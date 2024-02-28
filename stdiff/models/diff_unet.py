"""
Modified from https://github.com/WeilunWang/SinDiffusion/tree/main/guided_diffusion
"""
from abc import abstractmethod
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from torchdiffeq import odeint_adjoint as odeint
from torchsde import sdeint_adjoint as sdeint 
from omegaconf.omegaconf import open_dict

class DiffModel(nn.Module):
    def __init__(self, int_cfg, motion_encoder_cfg, diff_unet_cfg):
        super().__init__()
        self.sde = int_cfg.sde
        self.int_cfg = int_cfg
        self.motion_encoder_cfg = motion_encoder_cfg
        self.diff_unet_cfg = diff_unet_cfg

        self.learn_diff_image = self.motion_encoder_cfg.learn_diff_image
        n_downs = self.motion_encoder_cfg.n_downs
        image_size = self.motion_encoder_cfg.image_size
        H = W = int(image_size/(2**n_downs))
        motion_C = self.motion_encoder_cfg.model_channels*(2**n_downs)
        self.motion_feature_size = (motion_C, H, W)

        self.motion_encoder = MotionEncoder(self.motion_encoder_cfg)
        self.conv_gru_cell = ConvGRUCell(motion_C, motion_C, kernel_size = 3, stride = 1, padding = 1)
        
        #add missing configuration to the diff_unet_cfg
        with open_dict(self.diff_unet_cfg):
            self.diff_unet_cfg.image_size = H 
            self.diff_unet_cfg.in_channels = motion_C
            self.diff_unet_cfg.hidden_channels = motion_C

        self.diff_unet = DiffUnet(diff_unet_cfg, int_cfg, self.motion_feature_size)
        with open('debug.txt', 'a') as f:
            num_p_me, num_p_diff, num_p_gru = self.num_parameters()
            print(num_p_me, num_p_diff, num_p_gru, file = f)

    def context_encode(self, Vo, idx_o):
        """
        :param Vo: (N, To, C, H, W)
        :param idx_o: (To, )
        """
        if self.learn_diff_image:
            assert Vo.shape[1] >= 2, "invalid number of past frames"
            diff_images = Vo[:, 1:, ...] - Vo[:, 0:-1, ...] #(N, To-1, C, H, W)
            h = self.condition_enc(diff_images) #(N, To-1, C, H, W)
        else:
            #extract all the visual featurs of Vo by then encoder
            h = self.condition_enc(Vo)

        m = torch.zeros(self.motion_feature_size, device = h.device)
        m = repeat(m, 'C H W -> N C H W', N=Vo.shape[0])
        #update m given the first observed frame conditional feature
        m = self.conv_gru_cell(h[:, 0, ...], m)

        #recurrently calculate the context motion feature by GRU
        To = h.shape[1]
        for i in range(1, To):
            #integrate by ode/sde
            #m = odeint(self.diff_unet, m, idx_o[i-1:i+1])[-1, ...]
            m = self.conv_gru_cell(h[:, i, ...], m)

        return m
    
    def future_predict(self, m_context, idx_p):
        """
        :param m_context: (N, C, H, W)
        :param idx_p: (Tp+1, ), first element is the index of the motion context feature, i.e., last index of observed frames
        Return:
            m_future: (Tp, N, C, H, W)
        """
        
        if self.sde:
            N, C, H, W = m_context.shape
            m_future = sdeint(self.diff_unet, m_context.flatten(1), idx_p, 
                              method = self.int_cfg.method, dt=self.int_cfg.sde_options.dt,
                              rtol = self.int_cfg.sde_options.rtol, atol = self.int_cfg.sde_options.atol,
                              adaptive = self.int_cfg.sde_options.adaptive) #(t, N, C*H*W)
            m_future = rearrange(m_future, 't N (C H W) -> t N C H W', C = C, H = H, W = W)
        else:
            m_future = odeint(self.diff_unet, m_context, idx_p, method = self.int_cfg.method, options = self.int_cfg.ode_options)
            #m_future = odeint(self.diff_unet, m_context, idx_p, method = 'dopri5', rtol=1e-3, atol=1e-4)
        return m_future[1:, ...]
    
    def condition_enc(self, x):
        #x: (N, To, C, H, W)
        #encode the input x by a CNN encoder
        N, To, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.motion_encoder(x)
        
        return rearrange(x, '(N T) C H W -> N T C H W', N=N, T=To)
    
    def num_parameters(self):
        num_p_me = sum(p.numel() for p in self.motion_encoder.parameters() if p.requires_grad)
        num_p_diff = sum(p.numel() for p in self.diff_unet.parameters() if p.requires_grad)
        num_p_gru = sum(p.numel() for p in self.conv_gru_cell.parameters() if p.requires_grad)
        return num_p_me, num_p_diff, num_p_gru

class DiffUnet(nn.Module):
    def __init__(self, diff_unet_cfg, int_cfg, motion_feature_size):
        super().__init__()
        self.nonlienar = diff_unet_cfg.nonlinear
        self.n_layers = diff_unet_cfg.n_layers
        self.in_channels = diff_unet_cfg.in_channels
        self.out_channels = self.in_channels
        self.hidden_channels = diff_unet_cfg.hidden_channels
        
        self.int_cfg = int_cfg
        self.diff_unet_f = OdeSdeFuncNet(self.in_channels, self.hidden_channels, self.out_channels, self.n_layers, self.nonlienar)
        self.diff_unet_g = None
        self.sde = int_cfg.sde
        if self.sde:
            self.diff_unet_g = OdeSdeFuncNet(self.in_channels, self.hidden_channels, self.out_channels, self.n_layers, self.nonlienar)
            self.noise_type = self.int_cfg.sde_options.noise_type #must specify noise type and sde type for torchsde
            self.sde_type = self.int_cfg.sde_options.sde_type

        self.motion_feature_size = motion_feature_size
    
    def forward(self, t, x):
        """
        Only used for the ode
        """
        return self.diff_unet_f(x)
    
    def f(self, t, x):
        """
        only used for sde
        """
        C, H, W = self.motion_feature_size
        x = rearrange(x, 'N (C H W) -> N C H W', C = C, H = H, W = W)
        x = self.diff_unet_f(x)

        return x.flatten(1)

    def g(self, t, x):
        """
        only used for sde
        """
        C, H, W = self.motion_feature_size
        x = rearrange(x, 'N (C H W) -> N C H W', C = C, H = H, W = W)
        x = self.diff_unet_g(x)
        x = F.tanh(x)

        return x.flatten(1)

class OdeSdeFuncNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, nonlinear = 'Tanh'):
        super().__init__()
        if nonlinear == 'tanh':
            nonlinear_layer = nn.Tanh()
        else:
            raise NotImplementedError("Nonlinear layer is not supported")
        layers = []
        layers.append(nn.Conv2d(in_channels, hidden_channels, 3, 1, 1))
        for i in range(n_layers):
            layers.append(nonlinear_layer)
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1))
        layers.append(nonlinear_layer)
        layers.append(zero_module(nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        To Do: condition on diffusion time_steps
        """
        return self.net(x)

class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.GateConv = nn.Conv2d(in_channels+hidden_channels, 2*hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.NewStateConv = nn.Conv2d(in_channels+hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, inputs, prev_h):
        """
        inputs: (N, in_channels, H, W)
        Return:
            new_h: (N, hidden_channels, H, W)
        """
        gates = self.GateConv(torch.cat((inputs, prev_h), dim = 1))
        u, r = torch.split(gates, self.hidden_channels, dim = 1)
        u, r = F.sigmoid(u), F.sigmoid(r)
        h_tilde = F.tanh(self.NewStateConv(torch.cat((inputs, r*prev_h), dim = 1)))
        new_h = (1 - u)*prev_h + h_tilde

        return new_h


class MotionEncoder(nn.Module):
    """
    Modified from
    https://github.com/sunxm2357/mcnet_pytorch/blob/master/models/networks.py
    """
    def __init__(self, motion_encoder_cfg):
        super().__init__()

        input_dim=motion_encoder_cfg.in_channels
        ch=motion_encoder_cfg.model_channels
        n_downs=motion_encoder_cfg.n_downs
    
        model = []
        model += [nn.Conv2d(input_dim, ch, 5, padding = 2)]
        model += [nn.ReLU()]
        
        
        for _ in range(n_downs - 1):
            model += [nn.MaxPool2d(2)]
            model += [nn.Conv2d(ch, ch * 2, 5, padding = 2)]
            model += [nn.ReLU()]
            ch *= 2
        
        model += [nn.MaxPool2d(2)]
        model += [nn.Conv2d(ch, ch * 2, 7, padding = 3)]
        model += [nn.ReLU()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        x: (N, C, H, W)
        out: (N, C*(2^n_downs), H//(2^n_downs), W//(2^n_downs))
        """
        #TO DO: condition on diffferent diffusion timesteps
        out = self.model(x)
        return out
"""
#Spatial Fourier feature
class NRMLP(nn.Module):
    def __init__(self, out_channels, dim_x = 2, d_model = 64, MLP_layers = 4, scale = 10, fix_B = False):

        Modified based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
        The output layer is moved to the "PosFeatFuser"
  
        super().__init__()
        self.scale = scale
        self.dim_x = dim_x
        self.out_channels = out_channels
        self.MLP_layers = MLP_layers
        self.d_model = d_model
        self.fix_B = fix_B
        
        self.MLP = []

        self.mapping_fn = self.gaussian_mapping
        self.MLP.append(nn.Linear(2*self.d_model, self.d_model))
        if self.fix_B:
            self.register_buffer('B', torch.normal(mean = 0, std = 1.0, size = (self.d_model, self.dim_x)) * self.scale)
        else:
            #Default init for Linear is uniform distribution, would not produce a result as good as gaussian initialization
            #self.B = nn.Linear(self.dim_x, self.d_model, bias = False)
            self.B = nn.Parameter(torch.normal(mean = 0, std = 1.0, size = (self.d_model, self.dim_x)) * self.scale, requires_grad = True)
            
            #Init B with normal distribution or constant would produce much different result.
            #self.B = nn.Parameter(torch.ones(self.d_model, self.dim_x), requires_grad = True)
        
        self.MLP.append(nn.ReLU())
        for i in range(self.MLP_layers - 2):
            self.MLP.append(nn.Linear(self.d_model, self.d_model))
            self.MLP.append(nn.ReLU())
        
        self.MLP = nn.Sequential(*self.MLP)
        self.mlp_beta = nn.Linear(self.d_model, out_channels)

    def forward(self, x):

        Args:
            x: (N, d), N denotes the number of elements (coordinates)
        Return:
            out: (N, out_channels)

        x = self.mapping_fn(x)
        x = self.MLP(x)
        beta = self.mlp_beta(x)
        
        return beta
        

    def gaussian_mapping(self, x):

        Args:
            x: (N, d), N denotes the number of elements (coordinates)
            B: (m, d)

        proj = (2. * float(math.pi) * x) @ self.B.T
        #proj = self.B(2. * float(math.pi) * x)
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        return out


def CoorGenerator(h_list, w_list, max_H, max_W):

    The h/w/t index starts with 0
    Args:
        h_list: list of h coordinates, Tensor with shape (H,)
        w_list: list of w coordinates, Tensor with shape (W,)
    Returns:
        coor: Tensor with shape (H*W, 2), for the last dim, the coordinate order is (h, w)

    assert torch.max(h_list) <= max_H and torch.min(h_list) >= 0., "Invalid H coordinates"
    assert torch.max(w_list) <= max_W and torch.min(w_list) >= 0., "Invalid W coordinates"

    norm_h_list = h_list/max_H
    norm_w_list = w_list/max_W

    hvv, wvv = torch.meshgrid(norm_h_list, norm_w_list)
    s_coor = torch.stack([hvv, wvv], dim=-1)

    return s_coor.flatten(0, 1)
"""
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def create_diff_model(
    cfg
):
    motion_encoder_cfg = cfg.CDVP.DiffNet.MotionEncoder
    int_cfg = cfg.CDVP.DiffNet.Int
    diff_unet_cfg = cfg.CDVP.DiffNet.DiffUnet

    return DiffModel(int_cfg, motion_encoder_cfg, diff_unet_cfg)

if __name__ == '__main__':
    """
    enc = MotionEncoder(None)
    x = torch.randn(16, 3, 64, 64)
    print(enc(x, None).shape)
    """

    h_list, w_list = torch.linspace(0, 31, 32), torch.linspace(0, 31, 32)
    s_coor = CoorGenerator(h_list, w_list, 32, 32)
    
    print(s_coor.shape)