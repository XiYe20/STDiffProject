
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

from utils import visualize_batch_clips, MetricCalculator, AverageMeters, FVDFeatureExtractor
from pathlib import Path
from models import CDVPPipeline, CDVPDiffusers
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler, DDIMScheduler
from accelerate import Accelerator
from tqdm.auto import tqdm

def metric_test(Vp, pred, device):
    return 0


@hydra.main(version_base=None, config_path=".", config_name="config_test")
def main(cfg : DictConfig) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    ckpt_path = cfg.TestCfg.ckpt_path
    r_save_path = cfg.TestCfg.test_results_path
    if not Path(r_save_path).exists():
        Path(r_save_path).mkdir(parents=True, exist_ok=True) 
    
    mc = MetricCalculator(cfg.TestCfg.metrics).to(device)
    #ffe = FVDFeatureExtractor().to(device)
    if accelerator.is_main_process:
        ave_meter = AverageMeters(cfg.TestCfg.metrics)

    #load cdvp model
    cdvp = CDVPDiffusers.from_pretrained(ckpt_path, subfolder='cdvp').eval()
    #Print the number of parameters
    num_params = sum(p.numel() for p in cdvp.parameters() if p.requires_grad)
    print('Number of parameters is: ', num_params)

    #init scheduler
    if cfg.TestCfg.scheduler.name == 'DDPM':
        scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder = 'scheduler')
    elif cfg.TestCfg.scheduler.name == 'DPMMS':
        scheduler = DPMSolverMultistepScheduler.from_pretrained(ckpt_path, subfolder="scheduler", solver_order=3)
    else:
        raise NotImplementedError("Scheduler is not supported")

    cdvp_pipeline = CDVPPipeline(cdvp, scheduler).to(device)
    if not accelerator.is_main_process:
        cdvp_pipeline.disable_pgbar()
    _, _, test_loader = get_lightning_module_dataloader(cfg)
    #cdvp_pipeline, test_loader, mc, ffe = accelerator.prepare(cdvp_pipeline, test_loader, mc, ffe)
    cdvp_pipeline, test_loader, mc = accelerator.prepare(cdvp_pipeline, test_loader, mc)

    To = cfg.Dataset.test_num_observed_frames
    assert To == cfg.Dataset.num_observed_frames, 'invalid configuration'
    Tp = cfg.Dataset.test_num_predict_frames
    idx_o = torch.linspace(0, To-1 , To).to(device)
    if cfg.TestCfg.fps == 1:
        idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, cfg.Dataset.num_predict_frames).to(device)
    elif cfg.TestCfg.fps == 2:
        idx_p = torch.Tensor([2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]).to(device)
        #idx_p = torch.linspace(To, cfg.Dataset.num_predict_frames+To-1, 2*cfg.Dataset.num_predict_frames-1).to(device)
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
    def load_preds(obj_file):
        f = torch.load(obj_file)
        Vo_batch, Vp_batch, preds_batch = f['Vo'], f['g_Vp'], f['g_Preds']
        return Vo_batch, Vp_batch, preds_batch
    obj_file = '/Tmp/sifan/diffusion/CDVP_ckpts/cdvp_city_128_sde_autoreg/test_ddpm100_sample10_first_10/Preds_44.pt'
    Vo, Vp, _  = load_preds(obj_file)
    Vo = Vo*2. - 1.
    Vo = Vo[20:21, ...]
    Vp = Vp*2. - 1.
    Vp = Vp[20:21, ...]
    Vo, Vp = Vo.to(device), Vp.to(device)
    Vo_last_frame = Vo[:, -1:, ...]
    idx = 44
    print(Vo.min(), Vo.max())

    with torch.no_grad():
        progress_bar = tqdm(total=len(test_loader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Testing...") 

       #Vo, Vp, Vo_last_frame, _, _ = batch

        preds = []
        if cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
            filter_first_out = cdvp_pipeline.filter_best_first_pred(cfg.TestCfg.random_predict.first_pred_sample_num, Vo.clone(), 
                                                                    Vo_last_frame, Vp[:, 0:1, ...], idx_o, idx_p, 
                                                                    num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
                                                                    fix_init_noise=cfg.TestCfg.random_predict.fix_init_noise,
                                                                    bs = cfg.TestCfg.random_predict.first_pred_parralle_bs)
        for i in range(cfg.TestCfg.random_predict.sample_num):
            pred_clip = []
            Vo_input = Vo.clone()
            for j in range(autoreg_iter):
                if j == 0 and cfg.TestCfg.random_predict.first_pred_sample_num >= 2:
                    temp_pred = cdvp_pipeline.pred_remainig_frames(*(filter_first_out + (cfg.TestCfg.random_predict.fix_init_noise,"pil", False)))
                else:
                    temp_pred = cdvp_pipeline(Vo_input, Vo_last_frame, idx_o, idx_p, num_inference_steps = cfg.TestCfg.scheduler.sample_steps,
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

        #fvd_feat_Vp = ffe(Vp)
        #fvd_feat_preds = ffe(preds.detach().flatten(0, 1))

        #metric_dict = mc(Vp, preds)
        #g_metric_dict = accelerator.gather(metric_dict)
        #g_ffeat_Vp = accelerator.gather(fvd_feat_Vp)
        #g_ffeat_preds = accelerator.gather(fvd_feat_preds)

        g_preds = accelerator.gather(preds)
        g_Vo = accelerator.gather(Vo)
        g_Vp = accelerator.gather(Vp)
        print('g_preds shape', g_preds.shape)
        if accelerator.is_main_process:
            #ave_meter.iter_update(g_metric_dict)

            dump_obj = {'Vo': g_Vo.detach().cpu(), 'g_Vp': g_Vp.detach().cpu(), 'g_Preds': g_preds.detach().cpu()}
            torch.save(dump_obj, f=Path(r_save_path).joinpath(f'Preds_{idx}.pt'))
            #ave_meter.log_meter(Path(r_save_path))
            progress_bar.update(1)
            for i  in range(min(cfg.TestCfg.random_predict.sample_num, 4)):
                visualize_batch_clips(Vo, Vp, preds[:, i, ...], file_dir=Path(r_save_path).joinpath(f'test_examples_{idx}_traj{i}'))
        
            #del g_metric_dict
            #del g_ffeat_preds
            #del g_ffeat_Vp
            del g_Vo
            del g_Vp
            del g_preds
    print("Test finished")
if __name__ == '__main__':
    main()