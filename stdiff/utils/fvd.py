import torch
import os
import math
import os.path as osp
import math
import torch.nn.functional as F
from pathlib import Path
import scipy

# try:
#     from torchvision.models.utils import load_state_dict_from_url
# except ImportError:
#     from torch.utils.model_zoo import load_url as load_state_dict_from_url

# i3D_WEIGHTS_URL = "https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI"
"""
def load_i3d_pretrained(device=torch.device('cpu')):
    from .pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)

    filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_pretrained_400.pt')

    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d = torch.nn.DataParallel(i3d)
    i3d.eval()
    return i3d
"""

# https://github.com/universome/fvd-comparison
i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"


def load_i3d_pretrained(device=torch.device('cpu')):
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'i3d_torchscript.pt')
    if not os.path.exists(filepath):
        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    i3d = torch.jit.load(filepath).eval()
    #i3d = torch.nn.DataParallel(i3d)
    return i3d


def get_feats(videos, detector, device=None, bs=10):
    # videos : torch.tensor BCTHW [0, 1]
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    #detector_kwargs = dict()
    feats = np.empty((0, 400))
    #device = torch.device("cuda:0") if device is not torch.device("cpu") else device
    with torch.no_grad():
        for i in range((len(videos)-1)//bs + 1):
            temp = torch.stack([preprocess_single(video) for video in videos[i*bs:(i+1)*bs]])
            feats = np.vstack([feats, detector(temp, **detector_kwargs).detach().cpu().numpy()])
    return feats


def get_fvd_feats(videos, i3d, device=None, bs=10):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_feats(videos, i3d, device, bs)
    return embeddings

# """
# Copy-pasted from Copy-pasted from https://github.com/NVlabs/stylegan2-ada-pytorch
# """

# import ctypes
# import fnmatch
# import importlib
# import inspect
# import numpy as np
# import os
# import shutil
# import sys
# import types
# import io
# import pickle
# import re
# import requests
# import html
# import hashlib
# import glob
# import tempfile
# import urllib
# import urllib.request
# import uuid

# from distutils.util import strtobool
# from typing import Any, List, Tuple, Union, Dict

# def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
#     """Download the given URL and return a binary-mode file object to access the data."""
#     assert num_attempts >= 1

#     # Doesn't look like an URL scheme so interpret it as a local filename.
#     if not re.match('^[a-z]+://', url):
#         return url if return_filename else open(url, "rb")

#     # Handle file URLs.  This code handles unusual file:// patterns that
#     # arise on Windows:
#     #
#     # file:///c:/foo.txt
#     #
#     # which would translate to a local '/c:/foo.txt' filename that's
#     # invalid.  Drop the forward slash for such pathnames.
#     #
#     # If you touch this code path, you should test it on both Linux and
#     # Windows.
#     #
#     # Some internet resources suggest using urllib.request.url2pathname() but
#     # but that converts forward slashes to backslashes and this causes
#     # its own set of problems.
#     if url.startswith('file://'):
#         filename = urllib.parse.urlparse(url).path
#         if re.match(r'^/[a-zA-Z]:', filename):
#             filename = filename[1:]
#         return filename if return_filename else open(filename, "rb")

#     url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

#     # Download.
#     url_name = None
#     url_data = None
#     with requests.Session() as session:
#         if verbose:
#             print("Downloading %s ..." % url, end="", flush=True)
#         for attempts_left in reversed(range(num_attempts)):
#             try:
#                 with session.get(url) as res:
#                     res.raise_for_status()
#                     if len(res.content) == 0:
#                         raise IOError("No data received")

#                     if len(res.content) < 8192:
#                         content_str = res.content.decode("utf-8")
#                         if "download_warning" in res.headers.get("Set-Cookie", ""):
#                             links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
#                             if len(links) == 1:
#                                 url = requests.compat.urljoin(url, links[0])
#                                 raise IOError("Google Drive virus checker nag")
#                         if "Google Drive - Quota exceeded" in content_str:
#                             raise IOError("Google Drive download quota exceeded -- please try again later")

#                     match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
#                     url_name = match[1] if match else url
#                     url_data = res.content
#                     if verbose:
#                         print(" done")
#                     break
#             except KeyboardInterrupt:
#                 raise
#             except:
#                 if not attempts_left:
#                     if verbose:
#                         print(" failed")
#                     raise
#                 if verbose:
#                     print(".", end="", flush=True)

#     # Return data as file object.
#     assert not return_filename
#     return io.BytesIO(url_data)


def preprocess_single(video, resolution=224, sequence_length=None):
    # video: CTHW, [0, 1]
    c, t, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:, :sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

    # center crop
    c, t, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2
    return video.contiguous()


def get_logits(i3d, videos, device):
    #assert videos.shape[0] % 2 == 0
    logits = torch.empty(0, 400)
    with torch.no_grad():
        for i in range(len(videos)):
            # logits.append(i3d(preprocess_single(videos[i]).unsqueeze(0).to(device)).detach().cpu())
            logits = torch.vstack([logits, i3d(preprocess_single(videos[i]).unsqueeze(0).to(device)).detach().cpu()])
    # logits = torch.cat(logits, dim=0)
    return logits


def get_fvd_logits(videos, i3d, device):
    # videos in [0, 1] as torch tensor BCTHW
    # videos = [preprocess_single(video) for video in videos]
    embeddings = get_logits(i3d, videos, device)
    return embeddings


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L161
def _symmetric_matrix_square_root(mat, eps=1e-10):
    u, s, v = torch.linalg.svd(mat)
    si = torch.where(s < eps, s, torch.sqrt(s))
    return torch.matmul(torch.matmul(u, torch.diag(si)), v.t())


# https://github.com/tensorflow/gan/blob/de4b8da3853058ea380a6152bd3bd454013bf619/tensorflow_gan/python/eval/classifier_metrics.py#L400
def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = torch.matmul(sqrt_sigma, torch.matmul(sigma_v, sqrt_sigma))
    return torch.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()

    fact = 1.0 / (m.size(1) - 1) # unbiased estimate
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# def frechet_distance(x1, x2):
#     x1 = x1.flatten(start_dim=1)
#     x2 = x2.flatten(start_dim=1)
#     m, m_w = x1.mean(dim=0), x2.mean(dim=0)
#     sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
#     sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
#     trace = torch.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
#     mean = torch.sum((m - m_w) ** 2)
#     fd = trace + mean
#     return fd


"""
Copy-pasted from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
from scipy.linalg import sqrtm
import numpy as np


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0) # [d]
    sigma = np.cov(feats, rowvar=False) # [d, d]
    return mu, sigma


def frechet_distance(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    m = np.square(mu_gen - mu_real).sum()
    s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#https://github.com/NVlabs/long-video-gan/blob/f190c127525e665753c053e56e3e1d9091458bc8/metrics/metric_utils.py
class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None
        self.weight = 0.0

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x, weight=None):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        
        if weight is not None:
            assert weight.ndim == 1
            assert weight.shape[0] == x.shape[0]
                
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            
            if weight is not None:
                weight = weight[:self.max_items - self.num_items]

            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            if weight is None:
                self.raw_mean += x64.sum(axis=0)
                self.raw_cov += x64.T @ x64
                self.weight += x.shape[0]
            else:
                weight = weight.astype(np.float64)
                weighted_x64 = x64 * weight[:, None]
                self.raw_mean += weighted_x64.sum(axis=0)
                self.raw_cov += x64.T @ weighted_x64
                self.weight += weight.sum(axis=0)

    def append_torch(self, x, weight=None):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        x = x.cpu().numpy()
        weight = None if weight is None else gather_interleave(weight).cpu().numpy()
        self.append(x, weight)

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.weight
        cov = self.raw_cov / self.weight
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


def compute_feature_stats(ckpt_path):
    #calculate the max items (number of examples) of gt features and generated features
    fvd_feats = sorted(list(Path(ckpt_path).glob("FVD_feats*")))
    gt_num = 0
    pred_num = 0
    for file in fvd_feats:
        f = torch.load(file)
        gt, pred = f['ffeat_Vp'], f['ffeat_Preds']
        gt_num += gt.shape[0]
        pred_num += pred.shape[0]

    num_features = gt.shape[1]
    print(gt_num, pred_num, num_features)

    #gt_stats = FeatureStats(capture_mean_cov=True, max_items=gt_num)
    #pred_stats = FeatureStats(capture_mean_cov=True, max_items=pred_num)

    gt_feats = []
    pred_feats = []
    for file in fvd_feats:
        f = torch.load(file)
        gt, pred = f['ffeat_Vp'], f['ffeat_Preds']
        gt_feats.append(gt.cpu().numpy())
        pred_feats.append(pred.cpu().numpy())
        #gt_stats.append_torch(gt)
        #pred_stats.append_torch(pred)
    gt_feats = np.concatenate(gt_feats)
    pred_feats = np.concatenate(pred_feats)
    print(gt_feats.shape, pred_feats.shape)
    """
    mu_real, sigma_real = gt_stats.get_mean_cov()
    mu_gen, sigma_gen = pred_stats.get_mean_cov()
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    print(fid)
    """
    fid = frechet_distance(pred_feats, gt_feats)
    print(fid)

    return float(fid)

if __name__ == '__main__':
    compute_feature_stats('/Tmp/sifan/diffusion/CDVP_ckpts/cdvp_smmnist64_sde_autoreg/test_dpmms35_sample10')