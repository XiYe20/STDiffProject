
from utils import LitDataModule

from utils import get_lightning_module_dataloader
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import torch
import torch.nn as nn

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, omegaconf
from einops import rearrange

from utils import visualize_batch_clips, eval_metrics
from pathlib import Path
import argparse
from models import STDiffPipeline, STDiffDiffusers
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--test_config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    return args.test_config

def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True) 

    #load stdiff model
    stdiff = STDiffDiffusers.from_pretrained(ckpt_path, subfolder='stdiff').eval()
    #Print the number of parameters
    num_params = sum(p.numel() for p in stdiff.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    #init scheduler
    if cfg.TestCfg.scheduler.name == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')
    elif cfg.TestCfg.scheduler.name == 'DPMMS':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(ckpt_path, subfolder="scheduler", solver_order=3)
    else:
        raise NotImplementedError("Scheduler is not supported")

    stdiff_pipeline = STDiffPipeline(stdiff, scheduler).to(device)
    if not accelerator.is_main_process:
        stdiff_pipeline.disable_pgbar()
    _, _, test_loader = get_lightning_module_dataloader(cfg)
    stdiff_pipeline, test_loader = accelerator.prepare(stdiff_pipeline, test_loader)

    To = cfg.Dataset.test_num_observed_frames
    assert To == cfg.Dataset.num_observed_frames, 'invalid configuration'
    Tp = cfg.Dataset.test_num_predict_frames
    idx_o = torch.linspace(0, To-1 , To).to(device)
    if cfg.TestCfg.fps == 1:
        idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, cfg.Dataset.num_predict_frames).to(device)
    elif cfg.TestCfg.fps == 2:
        idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, 2*cfg.Dataset.num_predict_frames-1).to(device)
    #steps = cfg.TestCfg.fps*(cfg.Dataset.num_predict_frames-1) + 1
    #idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, steps).to(device)

    autoreg_iter = cfg.Dataset.test_num_predict_frames // cfg.Dataset.num_predict_frames
    autoreg_rem = cfg.Dataset.test_num_predict_frames % cfg.Dataset.num_predict_frames
    if autoreg_rem > 0:
        autoreg_iter = autoreg_iter + 1

    if accelerator.is_main_process:
        print('idx_o', idx_o)
        print('idx_p', idx_p)
        test_config = {'cfg': cfg, 'idx_o': idx_o.to('cpu'), 'idx_p': idx_p.to('cpu')}
        torch.save(test_config, f = Path(r_save_path).joinpath('TestConfig.pt'))
    
    def get_resume_batch_idx(r_save_path):
        save_path = Path(r_save_path)
        saved_preds = sorted(list(save_path.glob('Preds_*')))
        saved_batches = sorted([int(str(p.name).split('_')[1].split('.')[0]) for p in saved_preds])
        try:
            return saved_batches[-1]
        except IndexError:
            return -1
    resume_batch_idx = get_resume_batch_idx(r_save_path)
    print('number of test batches: ', len(test_loader))
    print('resume batch index: ', resume_batch_idx)

    #Predict and save the predictions to disk for evaluation
    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Testing...") 
        for idx, batch in enumerate(test_loader):
            if idx > resume_batch_idx: #resume test
                Vo, Vp, Vo_last_frame, _, _ = batch

                preds = []
                if cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
                    filter_first_out = stdiff_pipeline.filter_best_first_pred(cfg.TestCfg.random_predict.first_pred_sample_num, Vo.clone(), 
                                                                            Vo_last_frame, Vp[:, 0:1, ...], idx_o, idx_p, 
                                                                            num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                                                                            fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                                                                            bs = cfg.TestCfg.random_predict.first_pred_parralle_bs)
                for i in range(cfg.TestCfg.random_predict.sample_num):
                    pred_clip = []
                    Vo_input = Vo.clone()
                    for j in range(autoreg_iter):
                        if j == 0 and cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
                            temp_pred = stdiff_pipeline.pred_remainig_frames(*(filter_first_out + (cfg.TestCfg.random_predict.fix_init_noise,"pil", False)))
                        else:
                            temp_pred = stdiff_pipeline(Vo_input, Vo_last_frame, idx_o, idx_p, num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                                                    to_cpu=False, fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise) #Torch Tensor (N, Tp, C, H, W), range (0, 1)
                        pred_clip.append(temp_pred)
                        Vo_input = temp_pred[:, -To:, ...]*2. - 1.
                        Vo_last_frame = temp_pred[:, -1:, ...]*2. -1.

                    pred_clip = torch.cat(pred_clip, dim = 1)
                    if autoreg_rem > 0:
                        pred_clip = pred_clip[:, 0:(autoreg_rem - cfg.Dataset.num_predict_frames), ...]
                    preds.append(pred_clip)
                    
                preds = torch.stack(preds, 0) #(sample_num, N, Tp, C, H, W)
                preds = preds.permute(1, 0, 2, 3, 4, 5).contiguous() #(N, sample_num, num_predict_frames, C, H, W)
                Vo = (Vo / 2 + 0.5).clamp(0, 1)
                Vp = (Vp / 2 + 0.5).clamp(0, 1)

                g_preds = accelerator.gather(preds)
                g_Vo = accelerator.gather(Vo)
                g_Vp = accelerator.gather(Vp)

                if accelerator.is_main_process:
                    dump_obj = {'Vo': g_Vo.detach().cpu(), 'g_Vp': g_Vp.detach().cpu(), 'g_Preds': g_preds.detach().cpu()}
                    torch.save(dump_obj, f=Path(r_save_path).joinpath(f'Preds_{idx}.pt'))
                    progress_bar.update(1)
                    for i  in range(min(cfg.TestCfg.random_predict.sample_num, 4)):
                        visualize_batch_clips(Vo, Vp, preds[:, i, ...], file_dir=Path(r_save_path).joinpath(f'test_examples_{idx}_traj{i}'))

                    del g_Vo
                    del g_Vp
                    del g_preds
    print("Inference finished")
    print("Start evaluation metrics")
if __name__ == '__main__':
    config_path = Path(parse_args())
    initialize(version_base=None, config_path=str(config_path.parent))
    cfg = compose(config_name=str(config_path.name))

    main(cfg)
    eval_metrics(cfg)