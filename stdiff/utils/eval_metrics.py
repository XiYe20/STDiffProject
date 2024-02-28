
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

from utils import visualize_batch_clips, MetricCalculator, AverageMeters, FVDFeatureExtractor, frechet_distance
from pathlib import Path
from accelerate import Accelerator
from tqdm.auto import tqdm
import scipy.stats as st
import numpy as np

def eval_metrics(cfg : DictConfig) -> None:
    device = 'cuda:0'
    r_save_path = cfg.TestCfg.test_results_path
    
    mc = MetricCalculator(cfg.TestCfg.metrics).to(device)
    ave_meter = AverageMeters(cfg.TestCfg.metrics) #for ssim, psnr, lpips
    ffe = FVDFeatureExtractor().to(device)

    preds_files = sorted(list(Path(r_save_path).glob("Preds_*")))
    real_embeddings = []
    fake_embeddings = []
    with torch.no_grad():
        for file in tqdm(preds_files):
            f = torch.load(file)
            Vo_batch, Vp_batch, preds_batch = f['Vo'], f['g_Vp'], f['g_Preds']
            Vp_batch = Vp_batch[:, 0:17, ...]
            preds_batch = preds_batch[:, :, 0:17, ...]
            print(Vo_batch.shape, Vp_batch.shape, preds_batch.shape)

            bs = 16
            num_iter = Vo_batch.shape[0]//bs + (Vo_batch.shape[0]%bs != 0)
            for i in range(num_iter):
                if i == num_iter - 1:
                    Vo, Vp, preds = [load_b[i*bs:, ...].to(device) for load_b in (Vo_batch, Vp_batch, preds_batch)]
                else:
                    Vo, Vp, preds = [load_b[i*bs:(i+1)*bs, ...].to(device) for load_b in (Vo_batch, Vp_batch, preds_batch)]
                metric_dict = mc(Vp, preds)
                ave_meter.iter_update(metric_dict)

                real_fvd_embedding = ffe(torch.cat([Vo, Vp], dim = 1), return_numpy=True)
                N, num_sample = preds.shape[0], preds.shape[1]
                fake_fvd_embedding = ffe(torch.cat([Vo.unsqueeze(1).repeat(1, num_sample, 1, 1, 1, 1), preds], dim=2).flatten(0, 1), return_numpy=True).reshape(N, num_sample, -1)
                real_embeddings.append(real_fvd_embedding)
                fake_embeddings.append(fake_fvd_embedding)

    #calculate the fvd score
    real_embeddings = np.concatenate(real_embeddings, axis = 0)
    fake_embeddings = np.concatenate(fake_embeddings, axis = 0)

    fvds_list = []
    avg_fvd = frechet_distance(fake_embeddings.reshape(fake_embeddings.shape[0]*fake_embeddings.shape[1], -1), real_embeddings)
    for i in range(fake_embeddings.shape[1]):
        fvd = frechet_distance(fake_embeddings[:, i, ...], real_embeddings)
        fvds_list.append(fvd)
    best_fvd = np.min(fvds_list)
    fvd_traj_mean, fvd_traj_std  = float(np.mean(fvds_list)), float(np.std(fvds_list))
    fvd_traj_conf95 = fvd_traj_mean - float(st.norm.interval(alpha=0.95, loc=fvd_traj_mean, scale=st.sem(fvds_list))[0])
    print(avg_fvd, fvd_traj_mean, fvd_traj_std, fvd_traj_conf95, best_fvd)
    ave_meter.log_meter(r_save_path, avg_fvd)

if __name__ == '__main__':
    main()