import torch
from torch.utils import data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor
import pytorch_lightning as pl

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random
from typing import Optional
from itertools import groupby
from operator import itemgetter
from functools import partial
import random
from einops import rearrange

import cv2


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len_train_loader = None
        self.len_val_loader = None
        self.len_test_loader = None
        self.img_size = cfg.Dataset.image_size

        self.norm_transform = lambda x: x * 2. - 1.

        if cfg.Dataset.name == 'KTH':
            self.train_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.val_person_ids = [5]
        
        if cfg.Dataset.name == 'KITTI':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'MNIST':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        if cfg.Dataset.name == 'SMMNIST':
            self.train_transform = self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'BAIR':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'CityScapes':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        if cfg.Dataset.name == 'Human36M':
            self.train_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((self.img_size, self.img_size)), VidToTensor(), self.norm_transform])
        
        o_resize = None
        p_resize = None
        vp_size = cfg.STDiff.Diffusion.unet_config.sample_size
        vo_size = cfg.STDiff.DiffNet.MotionEncoder.image_size
        if vp_size != self.img_size:
            p_resize = transforms.Resize(vp_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        if vo_size != self.img_size:
            o_resize = transforms.Resize(vo_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.collate_fn = partial(svrfcn, rand_Tp=cfg.Dataset.rand_Tp, rand_predict=cfg.Dataset.rand_predict, o_resize=o_resize, p_resize=p_resize, half_fps=cfg.Dataset.half_fps)

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTrainData = KTHDataset(self.cfg.Dataset.dir, transform = self.train_transform, train = True, val = True, 
                                          num_observed_frames= self.cfg.Dataset.num_observed_frames, num_predict_frames= self.cfg.Dataset.num_predict_frames,
                                          val_person_ids = self.val_person_ids)#, actions = ['walking_no_empty'])
                self.train_set, self.val_set = KTHTrainData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.train_transform, train = True, val = True,
                                                num_observed_frames= self.cfg.Dataset.num_observed_frames, num_predict_frames= self.cfg.Dataset.num_predict_frames
                                                )
                self.train_set, self.val_set = KITTITrainData()

            if self.cfg.Dataset.name == 'BAIR':
                BAIR_train_whole_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()
                train_val_ratio = 0.95
                BAIR_train_set_length = int(len(BAIR_train_whole_set) * train_val_ratio)
                BAIR_val_set_length = len(BAIR_train_whole_set) - BAIR_train_set_length
                self.train_set, self.val_set = random_split(BAIR_train_whole_set, [BAIR_train_set_length, BAIR_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            if self.cfg.Dataset.name == 'CityScapes':
                self.train_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()
                self.val_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('val'), self.train_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames,
                                                   )()

            if self.cfg.Dataset.name == 'MNIST':
                self.train_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-train.npz'), self.train_transformo)
                self.val_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-valid.npz'), self.train_transform)
            
            if self.cfg.Dataset.name == 'SMMNIST':
                self.train_set = StochasticMovingMNIST(True, Path(self.cfg.Dataset.dir), self.cfg.Dataset.num_observed_frames, self.cfg.Dataset.num_predict_frames, self.train_transform)
                train_val_ratio = 0.95
                SMMNIST_train_set_length = int(len(self.train_set) * train_val_ratio)
                SMMNIST_val_set_length = len(self.train_set) - SMMNIST_train_set_length
                self.train_set, self.val_set = random_split(self.train_set, [SMMNIST_train_set_length, SMMNIST_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            
            if self.cfg.Dataset.name == 'Human36M':
                self.train_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames)()
                self.val_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('valid'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.num_observed_frames, num_predict_frames = self.cfg.Dataset.num_predict_frames)()

            #Use all training dataset for the final training
            if self.cfg.Dataset.phase == 'deploy':
                self.train_set = ConcatDataset([self.train_set, self.val_set])

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.train_set, _ = random_split(self.train_set, [dev_set_size, len(self.train_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
                self.val_set, _ = random_split(self.val_set, [dev_set_size, len(self.val_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            
            self.len_train_loader = len(self.train_dataloader())
            self.len_val_loader = len(self.val_dataloader())

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTestData = KTHDataset(self.cfg.Dataset.dir, transform = self.test_transform, train = False, val = False, 
                                        num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames)#, actions = ['walking_no_empty'])
                self.test_set = KTHTestData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.test_transform, train = False, val = False,
                                                num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames,
                                                )
                self.test_set = KITTITrainData()

            if self.cfg.Dataset.name == 'BAIR':
                self.test_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.test_transform, color_mode = 'RGB', 
                                            num_observed_frames= self.cfg.Dataset.test_num_observed_frames, num_predict_frames= self.cfg.Dataset.test_num_predict_frames, )()
            if self.cfg.Dataset.name == 'CityScapes':
                self.test_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.test_transform, color_mode = 'RGB', 
                                                   num_observed_frames = self.cfg.Dataset.test_num_observed_frames, num_predict_frames = self.cfg.Dataset.test_num_predict_frames,
                                                   )()
            if self.cfg.Dataset.name == 'MNIST':
                self.test_set = MovingMNISTDataset(Path(self.cfg.Dataset.dir).joinpath('moving-mnist-test.npz'), self.test_transform)
            
            if self.cfg.Dataset.name == 'SMMNIST':
                self.test_set = StochasticMovingMNIST(False, Path(self.cfg.Dataset.dir), self.cfg.Dataset.test_num_observed_frames, self.cfg.Dataset.test_num_predict_frames, self.test_transform)
            
            if self.cfg.Dataset.name == 'Human36M':
                self.test_set = Human36MDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.train_transform, color_mode = 'RGB', num_observed_frames = self.cfg.Dataset.test_num_observed_frames, num_predict_frames = self.cfg.Dataset.test_num_predict_frames)()

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.test_set, _ = random_split(self.test_set, [dev_set_size, len(self.test_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            self.len_test_loader = len(self.test_dataloader())

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle = False, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = False, collate_fn = self.collate_fn)


def get_lightning_module_dataloader(cfg):
    pl_datamodule = LitDataModule(cfg)
    pl_datamodule.setup()
    return pl_datamodule.train_dataloader(), pl_datamodule.val_dataloader(), pl_datamodule.test_dataloader()

class KTHDataset(object):
    """
    KTH dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (120, 160)
    Split the KTH dataset and return the train and test dataset
    """
    def __init__(self, KTH_dir, transform, train, val,
                 num_observed_frames, num_predict_frames, actions=['boxing', 'handclapping', 'handwaving', 'jogging_no_empty', 'running_no_empty', 'walking_no_empty'], val_person_ids = None
                 ):
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.KTH_path = Path(KTH_dir).absolute()
        self.actions = actions
        self.train = train
        self.val = val
        if self.train:
            self.person_ids = list(range(1, 17))
            if self.val:
                if val_person_ids is None: #one person for the validation
                    self.val_person_ids = [random.randint(1, 17)]
                    self.person_ids.remove(self.val_person_ids[0])
                else:
                    self.val_person_ids = val_person_ids
        else:
            self.person_ids = list(range(17, 26))

        frame_folders = self.__getFramesFolder__(self.person_ids)
        self.clips = self.__getClips__(frame_folders)
        
        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_person_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        
        clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform, self.color_mode)
        if self.val:
            val_clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.val_clips, self.transform, self.color_mode)
            return clip_set, val_clip_set
        else:
            return clip_set
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
    def __getFramesFolder__(self, person_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.actions:
            action_path = self.KTH_path.joinpath(a)
            frame_folders.extend([action_path.joinpath(s) for s in os.listdir(action_path) if '.avi' not in s])
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            person_id = int(str(ff.name).strip().split('_')[0][-2:])
            if person_id in person_ids:
                return_folders.append(ff)

        return return_folders

class BAIRDataset(object):
    """
    BAIR dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (64, 64)
    The train and test frames has been previously splitted: ref "Self-Supervised Visual Planning with Temporal Skip Connections"
    """
    def __init__(self, frames_dir: str, transform, color_mode = 'RGB', 
                 num_observed_frames = 10, num_predict_frames = 10):
        """
        Args:
            frames_dir --- Directory of extracted video frames and original videos.
            transform --- trochvison transform functions
            color_mode --- 'RGB' or 'grey_scale' color mode for the dataset
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clip_length --- number of frames for each video clip example for model
        """
        self.frames_path = Path(frames_dir).absolute()
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = color_mode

        self.clips = self.__getClips__()


    def __call__(self):
        """
        Returns:
            data_set --- ClipDataset object
        """
        data_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.clips, self.transform, self.color_mode)

        return data_set
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class CityScapesDataset(BAIRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            all_imgs = sorted(list(folder.glob('*')))
            obj_dict = {}
            for f in all_imgs:
                id = str(f).split('_')[1]
                if id in obj_dict:
                    obj_dict[id].append(f)
                else:
                    obj_dict[id] = [f]
            for k, img_files in obj_dict.items():
                for k, g in groupby(enumerate(img_files), lambda ix: ix[0]-int(str(ix[1]).split('_')[2])):
                    clip_files = list(list(zip(*list(g)))[1])
                    
                    clip_num = len(clip_files) // self.clip_length
                    rem_num = len(clip_files) % self.clip_length
                    clip_files = clip_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
                    for i in range(clip_num):
                        clips.append(clip_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class KITTIDataset(object):
    def __init__(self, KITTI_dir, test_folder_ids, transform, train, val,
                 num_observed_frames, num_predict_frames):
        """
        Args:
            KITTI_dir --- Directory for extracted KITTI video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clip_length = num_observed_frames + num_predict_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.KITTI_path = Path(KITTI_dir).absolute()
        self.train = train
        self.val = val

        self.all_folders = sorted(os.listdir(self.KITTI_path))
        self.num_examples = len(self.all_folders)
        
        self.folder_id = list(range(self.num_examples))
        if self.train:
            self.train_folders = [self.all_folders[i] for i in range(self.num_examples) if i not in test_folder_ids]
            if self.val:
                self.val_folders = self.train_folders[0:2]
                self.train_folders = self.train_folders[2:]
    
        else:
            self.test_folders = [self.all_folders[i] for i in test_folder_ids]
        
        if self.train:
            self.train_clips = self.__getClips__(self.train_folders)
            if self.val:
                self.val_clips = self.__getClips__(self.val_folders)
        else:
            self.test_clips = self.__getClips__(self.test_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        if self.train:
            clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.train_clips, self.transform, self.color_mode)
            if self.val:
                val_clip_set = ClipDataset(self.num_observed_frames, self.num_predict_frames, self.val_clips, self.transform, self.color_mode)
                return clip_set, val_clip_set
            return clip_set
        else:
            return ClipDataset(self.num_observed_frames, self.num_predict_frames, self.test_clips, self.transform, self.color_mode)
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(self.KITTI_path.joinpath(folder).glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
class ClipDataset(Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_observed_frames, num_predict_frames, clips, transform, color_mode):
        """
        Args:
            num_observed_frames --- number of past frames
            num_predict_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_observed_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_predict_frames, C, H, W)
        """
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.clips = clips
        self.transform = transform
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_observed_frames, C, H, W)
            future_clip: Tensor with shape (num_predict_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_imgs = self.clips[index]
        imgs = []
        for img_path in clip_imgs:
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
        
        original_clip = self.transform(imgs)

        past_clip = original_clip[0:self.num_observed_frames, ...]
        future_clip = original_clip[-self.num_predict_frames:, ...]
        return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)
        
        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
        video = cv2.VideoWriter(Path(file_name).absolute().as_posix(), fourcc, 10, videodims)
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        #imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])

class StochasticMovingMNIST(Dataset):
    """https://github.com/edenton/svg/blob/master/data/moving_mnist.py"""
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, train_flag, data_root, num_observed_frames, num_predict_frames, transform, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.num_observed_frames = num_observed_frames
        self.num_predict_frames = num_predict_frames
        self.seq_len = num_observed_frames + num_predict_frames
        self.transform = transform
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train_flag,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size, antialias=True),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        full_clip = torch.from_numpy(self.__getnparray__(idx))
        imgs = []
        for i in range(full_clip.shape[0]):
            img = transforms.ToPILImage()(full_clip[i, ...])
            imgs.append(img)
        
        full_clip = self.transform(imgs)

        past_clip = full_clip[0:self.num_observed_frames, ...]
        future_clip = full_clip[self.num_observed_frames:, ...]

        return past_clip, future_clip

    def __getnparray__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x.transpose(0, 3, 1, 2)

def svrfcn(batch_data, rand_Tp = 3, rand_predict = True, o_resize = None, p_resize = None, half_fps = False):
    """
    Single video dataset random future frames collate function
    batch_data: list of tuples, each tuple is (observe_clip, predict_clip)
    """
    
    observe_clips, predict_clips = zip(*batch_data)
    observe_batch = torch.stack(observe_clips, dim=0)
    predict_batch = torch.stack(predict_clips, dim=0)

    #output the last frame of observation, taken as the first frame of autoregressive prediction
    observe_last_batch = observe_batch[:, -1:, ...]
    
    max_Tp = predict_batch.shape[1]
    if rand_predict:
        assert rand_Tp <= max_Tp, "Invalid rand_Tp"
        rand_idx = np.sort(np.random.choice(max_Tp, rand_Tp, replace=False))
        rand_idx = torch.from_numpy(rand_idx)
        rand_predict_batch = predict_batch[:, rand_idx, ...]
    else:
        rand_idx = torch.linspace(0, max_Tp-1, max_Tp, dtype = torch.int)
        rand_predict_batch = predict_batch
    To = observe_batch.shape[1]
    idx_o = torch.linspace(0, To-1 , To, dtype = torch.int)

    if half_fps:
        if observe_batch.shape[1] > 2:
            observe_batch = observe_batch[:, ::2, ...]
            idx_o = idx_o[::2, ...]

        rand_predict_batch = rand_predict_batch[:, ::2, ...]
        rand_idx = rand_idx[::2, ...]
        observe_last_batch = observe_batch[:, -1:, ...]

    if p_resize is not None:
        N, T, _, _, _ = rand_predict_batch.shape
        rand_predict_batch = p_resize(rand_predict_batch.flatten(0, 1))
        rand_predict_batch = rearrange(rand_predict_batch, "(N T) C H W -> N T C H W", N = N, T=T)
        #als resize the last frame of observation
        observe_last_batch = p_resize(observe_last_batch.flatten(0, 1))
        observe_last_batch = rearrange(observe_last_batch, "(N T) C H W -> N T C H W", N = N, T=1)
        
    if o_resize is not None:
        N, T, _, _, _ = observe_batch.shape
        observe_batch = o_resize(observe_batch.flatten(0, 1))
        observe_batch = rearrange(observe_batch, "(N T) C H W -> N T C H W", N = N, T=T)
    return (observe_batch, rand_predict_batch, observe_last_batch, idx_o.to(torch.float), rand_idx.to(torch.float) + To)

#####################################################################################
class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.resize_kwargs['antialias'] = True
        self.resize_kwargs['interpolation'] = transforms.InterpolationMode.BICUBIC
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip

class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]

        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.Dataset:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.Dataset.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def visualize_batch_clips(gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch, file_dir, renorm_transform = None, desc = None):
    """
        pred_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_future_frames_batch: tensor with shape (N, future_clip_length, C, H, W)
        gt_past_frames_batch: tensor with shape (N, past_clip_length, C, H, W)
    """
    if not Path(file_dir).exists():
        Path(file_dir).mkdir(parents=True, exist_ok=True) 
    def save_clip(clip, file_name):
        imgs = []
        if renorm_transform is not None:
            clip = renorm_transform(clip)
            clip = torch.clamp(clip, min = 0., max = 1.0)
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:], loop = 0)
    
    def append_frames(batch, max_clip_length):
        d = max_clip_length - batch.shape[1]
        batch = torch.cat([batch, batch[:, -2:-1, :, :, :].repeat(1, d, 1, 1, 1)], dim = 1)
        return batch
    max_length = max(gt_future_frames_batch.shape[1], gt_past_frames_batch.shape[1])
    max_length = max(max_length, pred_frames_batch.shape[1])
    if gt_past_frames_batch.shape[1] < max_length:
        gt_past_frames_batch = append_frames(gt_past_frames_batch, max_length)
    if gt_future_frames_batch.shape[1] < max_length:
        gt_future_frames_batch = append_frames(gt_future_frames_batch, max_length)
    if pred_frames_batch.shape[1] < max_length:    
        pred_frames_batch = append_frames(pred_frames_batch, max_length)

    batch = torch.cat([gt_past_frames_batch, gt_future_frames_batch, pred_frames_batch], dim = -1) #shape (N, clip_length, C, H, 3W)
    batch = batch.cpu()
    N = batch.shape[0]
    for n in range(N):
        clip = batch[n, ...]
        file_name = file_dir.joinpath(f'{desc}_clip_{n}.gif')
        save_clip(clip, file_name)