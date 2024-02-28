import torch
import torchvision.transforms as transforms

from diffusers import ConfigMixin, ModelMixin, register_to_config, UNet2DMotionCond
from .diff_unet import DiffModel
from omegaconf import OmegaConf

class STDiffDiffusers(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, unet_cfg, tde_cfg):
        super().__init__()
        try:
            self.autoreg = tde_cfg.autoregressive
            self.super_res_training = tde_cfg.super_res_training
            self.tde_model = DiffModel(tde_cfg.Int, tde_cfg.MotionEncoder, tde_cfg.DiffUnet)
        except AttributeError:
            tde_cfg = OmegaConf.structured(tde_cfg)
            self.autoreg = tde_cfg.autoregressive
            self.super_res_training =  tde_cfg.super_res_training
            self.tde_model = DiffModel(tde_cfg.Int, tde_cfg.MotionEncoder, tde_cfg.DiffUnet)
        self.diffusion_unet = UNet2DMotionCond(**unet_cfg)

    def forward(self, Vo, idx_o, idx_p, noisy_Vp, timestep, clean_Vp = None, Vo_last_frame=None):
        #vo: (N, To, C, Ho, Wo), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, Hp, Wp)
        m_context = self.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)
            
        #use ode/sde to predict the future motion features
        m_future = self.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

        if self.autoreg:
            assert clean_Vp is not None and Vo_last_frame is not None, "input clean Vp and last frame of observation for autoregressive prediction."
            #for the superresolution model, prev_frames have a lower resolution (Ho, Wo)

            N, To, C, Ho, Wo = Vo.shape
            N, Tp, C, Hp, Wp = clean_Vp.shape
            if self.super_res_training:
                if Ho < Hp or Wo < Wp:
                    down_sample= transforms.Resize((Ho, Wo), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                    up_sample = transforms.Resize((Hp, Wp), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)

                    clean_Vp = up_sample(down_sample(clean_Vp.flatten(0, 1)))
                    clean_Vp = clean_Vp.reshape(N, Tp, C, Hp, Wp)
                    Vo_last_frame = up_sample(Vo[:, -1, ...]).reshape(N, 1, C, Hp, Wp)
            prev_frames = torch.cat([Vo_last_frame, clean_Vp[:, 0:-1, ...]], dim = 1)
            noisy_Vp = torch.cat([noisy_Vp, prev_frames.flatten(0, 1)], dim = 1)

        out = self.diffusion_unet(noisy_Vp, timestep, m_feat = m_future.permute(1, 0, 2, 3, 4).flatten(0, 1))
        
        return out