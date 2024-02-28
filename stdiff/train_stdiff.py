import inspect
import logging
import math
import os
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
import argparse

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available


import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from utils import get_lightning_module_dataloader
from models import STDiffDiffusers, STDiffPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--train_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.train_config

def main(cfg : DictConfig) -> None:
    logging_dir = os.path.join(cfg.Env.output_dir, 'logs')

    accelerator_project_config = ProjectConfiguration(total_limit=cfg.Training.epochs // cfg.Training.save_model_epochs)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.Training.gradient_accumulation_steps,
        mixed_precision=cfg.Training.mixed_precision,
        log_with=cfg.Env.logger,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if cfg.Env.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfg.Training.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfg.Training.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), STDiffDiffusers)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = STDiffDiffusers.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.Env.output_dir is not None:
            os.makedirs(cfg.Env.output_dir, exist_ok=True)

    # Initialize the model
    model = STDiffDiffusers(cfg.STDiff.Diffusion.unet_config, cfg.STDiff.DiffNet)
    num_p_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num params of stdiff: {num_p_model/1e6} M')

    if cfg.Env.stdiff_init_ckpt is not None:
        model = STDiffDiffusers.from_pretrained(cfg.Env.stdiff_init_ckpt, subfolder='unet')
        print('Init from a checkpoint')

    # Create EMA for the model.
    if cfg.Training.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=cfg.Training.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=cfg.Training.ema_inv_gamma,
            power=cfg.Training.ema_power,
            model_cls=STDiffDiffusers,
            model_config=model.config,
        )

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps,
            beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule,
            prediction_type=cfg.STDiff.Diffusion.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.STDiff.Diffusion.ddpm_num_steps, beta_schedule=cfg.STDiff.Diffusion.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.Training.learning_rate,
        betas=cfg.Training.adam_betas,
        weight_decay=cfg.Training.adam_weight_decay,
        eps=cfg.Training.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # Preprocessing the datasets and DataLoaders creation.
    train_dataloader, val_dataloader, test_dataloader = get_lightning_module_dataloader(cfg)

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.Training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.Training.lr_warmup_steps * cfg.Training.gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * cfg.Training.epochs,
        num_cycles=cfg.Training.num_cycles,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if cfg.Training.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = cfg.Dataset.batch_size * accelerator.num_processes * cfg.Training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.Training.gradient_accumulation_steps)
    max_train_steps = cfg.Training.epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {cfg.Training.epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.Dataset.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.Training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.Env.resume_ckpt is None:
        accelerator.print(
            f"Starting a new training run."
        )
        cfg.Env.resume_ckpt = None
    else:
        accelerator.print(f"Resuming from checkpoint {cfg.Env.resume_ckpt}")
        accelerator.load_state(os.path.join(cfg.Env.output_dir, cfg.Env.resume_ckpt))
        global_step = int(cfg.Env.resume_ckpt.split("-")[1])

        resume_global_step = global_step * cfg.Training.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * cfg.Training.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, cfg.Training.epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                Vo, Vp, Vo_last_frame, idx_o, idx_p = batch
            
                clean_images = Vp.flatten(0, 1)

                # Skip steps until we reach the resumed step
                if cfg.Env.resume_ckpt and epoch == first_epoch and step < resume_step:
                    if step % cfg.Training.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # Sample noise that we'll add to the images
                if not cfg.STDiff.DiffNet.autoregressive:
                    N, Tp, C, H, W = Vp.shape
                    noise = torch.randn(N, C, H, W).unsqueeze(1).repeat(1, Tp, 1, 1, 1).flatten(0, 1).to(clean_images.device)
                    Vo_last_frame = None

                else:
                    noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                # Predict the noise residual
                model_output = model(Vo, idx_o, idx_p, noisy_images, timesteps, Vp, Vo_last_frame).sample

                if cfg.STDiff.Diffusion.prediction_type == "epsilon":
                    loss = F.l1_loss(model_output, noise)  # this could have different weights!
                elif cfg.STDiff.Diffusion.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.l1_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {cfg.STDiff.Diffusion.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfg.Training.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.Training.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfg.Env.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if cfg.Training.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.Training.save_images_epochs == 0 or epoch == cfg.Training.epochs - 1:
                unet = accelerator.unwrap_model(model)

                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = STDiffPipeline(
                    stdiff=unet,
                    scheduler=noise_scheduler,
                )

                generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                Vo, _, Vo_last_frame, idx_o, idx_p = next(iter(train_dataloader))
                images = pipeline(
                    Vo,
                    Vo_last_frame,
                    idx_o,
                    idx_p,
                    generator=generator,
                    num_inference_steps=cfg.STDiff.Diffusion.ddpm_num_inference_steps,
                    output_type="numpy"
                ).images

                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if cfg.Env.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif cfg.Env.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % cfg.Training.save_model_epochs == 0 or epoch == cfg.Training.epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if cfg.Training.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = STDiffPipeline(
                    stdiff=unet,
                    scheduler=noise_scheduler
                )

                pipeline.save_pretrained(cfg.Env.output_dir)

                if cfg.Training.use_ema:
                    ema_model.restore(unet.parameters())

    accelerator.end_training()

if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)
    main()