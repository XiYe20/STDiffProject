# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from diffusers import utils
from diffusers import DiffusionPipeline, ImagePipelineOutput
import torchvision.transforms as transforms
from math import exp

from einops import rearrange

class STDiffPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, stdiff, scheduler):
        super().__init__()
        self.register_modules(stdiff=stdiff, scheduler=scheduler)
        
    @torch.no_grad()
    def __call__(
        self,
        Vo,
        Vo_last_frame,
        idx_o,
        idx_p,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        to_cpu=True,
        fix_init_noise=None
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        #set default value for fix_init_noise
        if fix_init_noise is None:
            if not self.stdiff.autoreg:
                fix_init_noise = True
            else:
                fix_init_noise = False

        if not self.stdiff.autoreg:
            # Sample gaussian noise to begin loop
            if fix_init_noise:
                image_shape = (Vo.shape[0], self.stdiff.diffusion_unet.in_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
            else:
                batch_size = Vo.shape[0]*idx_p.shape[0]
                image_shape = (batch_size, self.stdiff.diffusion_unet.in_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
                
            image = self.init_noise(image_shape, generator)
            if fix_init_noise:
                image = image.unsqueeze(1).repeat(1, idx_p.shape[0], 1, 1, 1).flatten(0, 1)
            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # manually extract the future motion feature
            #vo: (N, To, C, H, W), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, H, W)
            m_context = self.stdiff.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)
                
            #use ode/sde to predict the future motion features
            m_future = self.stdiff.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.stdiff.diffusion_unet(image, t, m_feat = m_future.permute(1, 0, 2, 3, 4).flatten(0, 1)).sample

                # 2. compute previous image: x_t -> x_t-1
                #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                image = self.scheduler.step(model_output, t, image).prev_sample
        
        else:
            # Sample gaussian noise to begin loop
            image_shape = (Vo.shape[0], self.stdiff.diffusion_unet.out_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
            
            # set step values
            self.scheduler.set_timesteps(num_inference_steps)

            # manually extract the future motion feature
            #vo: (N, To, C, H, W), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, H, W)
            m_context = self.stdiff.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)

            m_future = self.stdiff.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p])) #(Tp, N, C, H, W)

            #for the superresolution training
            Ho, Wo = Vo.shape[3], Vo.shape[4]
            Hp, Wp = image_shape[2], image_shape[3]
            down_sample = lambda x: x
            up_sample = lambda x: x
            if self.stdiff.super_res_training:
                if Ho < Hp or Wo < Wp:
                    down_sample= transforms.Resize((Ho, Wo), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
                    up_sample = transforms.Resize((Hp, Wp), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            
            image = self.init_noise(image_shape, generator)
            imgs = []
            for tp in range(idx_p.shape[0]):
                if tp == 0:
                    if self.stdiff.super_res_training:
                        Vo_last_frame = up_sample(Vo[:, -1, ...])
                        prev_frame = Vo_last_frame
                    else:
                        prev_frame = Vo_last_frame[:, -1, ...]
                else:
                    if self.stdiff.super_res_training:
                        prev_frame = up_sample(down_sample(imgs[-1]))
                    else:
                        prev_frame = imgs[-1]
                
                if not fix_init_noise:
                    image = self.init_noise(image_shape, generator)
                
                for t in self.progress_bar(self.scheduler.timesteps):
                    # 1. predict noise model_output
                    model_output = self.stdiff.diffusion_unet(torch.cat([image, prev_frame.clamp(-1, 1)], dim = 1), t, m_feat = m_future[tp, ...]).sample

                    # 2. compute previous image: x_t -> x_t-1
                    #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                    image = self.scheduler.step(model_output, t, image).prev_sample

                imgs.append(image)

            image = torch.stack(imgs, dim = 1).flatten(0, 1)
            
        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "numpy":
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return ImagePipelineOutput(images=image)
        else:
            image = rearrange(image, '(N T) C H W -> N T C H W', N = Vo.shape[0], T = idx_p.shape[0])
            if to_cpu:
                image = image.cpu()
            return image
    
    def init_noise(self, image_shape, generator):
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = utils.randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = utils.randn_tensor(image_shape, generator=generator, device=self.device)
        
        return image
    
    def disable_pgbar(self):
        self.progress_bar = lambda x: x

    def filter_best_first_pred(self, first_pred_sample_num, Vo, Vo_last_frame, Vp_first_frame, idx_o, idx_p, 
                               generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                               num_inference_steps: int = 1000,
                               fix_init_noise=False,
                               bs = 4):
        # Sample gaussian noise to begin loop
        image_shape = (Vo.shape[0], self.stdiff.diffusion_unet.out_channels, self.stdiff.diffusion_unet.sample_size, self.stdiff.diffusion_unet.sample_size)
        
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # manually extract the future motion feature
        #vo: (N, To, C, H, W), idx_o: (To, ), idx_p: (Tp, ), noisy_Vp: (N*Tp, C, H, W)
        m_context = self.stdiff.tde_model.context_encode(Vo, idx_o) #(N, C, H, W)

        assert Vp_first_frame is not None, "Please input ground truth first future frame"
        prev_frame = Vo_last_frame[:, -1, ...]
        m_first_all = []
        for i in range(first_pred_sample_num):
            m_first = self.stdiff.tde_model.future_predict(m_context, torch.cat([idx_o[-1:], idx_p[0:1]]))[0, ...] #(N, C, H, W)
            m_first_all.append(m_first)
        
        num_iter = first_pred_sample_num//bs + (first_pred_sample_num%bs != 0)
        first_preds = []
        for i in range(num_iter):
            m_first = torch.stack(m_first_all[i*bs:(i+1)*bs], dim = 1)
            rn = m_first.shape[1]
            image = self.init_noise(image_shape, generator)
            image = image.repeat(rn, 1, 1, 1)
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.stdiff.diffusion_unet(torch.cat([image, prev_frame.repeat(rn, 1, 1, 1)], dim = 1), t, m_feat = m_first.flatten(0, 1)).sample

                # 2. compute previous image: x_t -> x_t-1
                #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                image = self.scheduler.step(model_output, t, image).prev_sample
            first_preds.append(image)
            #first_preds.append(rearrange(image, '(N r) C H W -> N r C H W', N=m_first.shape[0], r=rn))
        first_preds = torch.cat(first_preds, dim = 0)
        
        #calculate the PSNR, SSIM of first frames
        ssim = SSIM()(Vp_first_frame[:, 0, ...].repeat(first_pred_sample_num, 1, 1, 1), first_preds, mean_flag=False)
        psnr = PSNR(Vp_first_frame[:, 0, ...].repeat(first_pred_sample_num, 1, 1, 1), first_preds, mean_flag=False)
        ssim = ssim.reshape(Vp_first_frame.shape[0], first_pred_sample_num)
        psnr = psnr.reshape(Vp_first_frame.shape[0], first_pred_sample_num)
        merge_metric = ssim+psnr
        best_idx = torch.argmax(merge_metric, dim = 1).cpu().numpy().tolist()
        def FlattenBestIdx(best_idx):
            N = len(best_idx)
            flatten_best_idx = []
            for n, idx in enumerate(best_idx):
                flatten_best_idx.append(N*idx+n)
            return flatten_best_idx
        flatten_best_idx = FlattenBestIdx(best_idx)
        best_first_preds = first_preds[flatten_best_idx, ...]

        m_first_all = torch.cat(m_first_all, dim = 0)
        best_m_first = m_first_all[flatten_best_idx]
        return best_m_first, best_first_preds, idx_p, image_shape, generator
    
    def pred_remainig_frames(self, best_m_first, best_first_preds, idx_p, image_shape, generator, fix_init_noise=False,
                             output_type: Optional[str] = "pil", to_cpu=True):
        m_future = self.stdiff.tde_model.future_predict(best_m_first, idx_p) #(Tp-1, N, C, H, W)
        image = self.init_noise(image_shape, generator)
        imgs = [best_first_preds]
        for tp in range(1, idx_p.shape[0]):
            if tp == 1:
                prev_frame = best_first_preds
            else:
                prev_frame = imgs[-1]
            
            if not fix_init_noise:
                image = self.init_noise(image_shape, generator)
            
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.stdiff.diffusion_unet(torch.cat([image, prev_frame.clamp(-1, 1)], dim = 1), t, m_feat = m_future[tp-1, ...]).sample

                # 2. compute previous image: x_t -> x_t-1
                #image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample
                image = self.scheduler.step(model_output, t, image).prev_sample

            imgs.append(image)

        image = torch.stack(imgs, dim = 1).flatten(0, 1)

        image = (image / 2 + 0.5).clamp(0, 1)
        if output_type == "numpy":
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            return ImagePipelineOutput(images=image)
        else:
            image = rearrange(image, '(N T) C H W -> N T C H W', N = best_first_preds.shape[0], T = idx_p.shape[0])
            if to_cpu:
                image = image.cpu()
            return image


def PSNR(x: Tensor, y: Tensor, data_range: Union[float, int] = 1.0, mean_flag: bool = True) -> Tensor:
    """
    Comput the average PSNR between two batch of images.
    x: input image, Tensor with shape (N, C, H, W)
    y: input image, Tensor with shape (N, C, H, W)
    data_range: the maximum pixel value range of input images, used to normalize
                pixel values to [0,1], default is 1.0
    """

    EPS = 1e-8
    x = x/float(data_range)
    y = y/float(data_range)

    mse = torch.mean((x-y)**2, dim = (1, 2, 3))
    score = -10*torch.log10(mse + EPS)
    if mean_flag:
        return torch.mean(score).item()
    else:
        return score

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        self.__name__ = 'SSIM'
        
    def forward(self, img1: Tensor, img2: Tensor, mean_flag: bool = True) -> float:
        """
        img1: (N, C, H, W)
        img2: (N, C, H, W)
        Return:
            batch average ssim_index: float
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return self._ssim(img1, img2, window, self.window_size, channel, mean_flag)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window

    def _ssim(self, img1, img2, window, window_size, channel, mean_flag):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if mean_flag:
            return ssim_map.mean()
        else:
            return torch.mean(ssim_map, dim=(1,2,3))