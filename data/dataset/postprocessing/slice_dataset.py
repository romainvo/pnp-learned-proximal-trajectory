from typing import Optional, Mapping, Tuple, Union, Any, List, Callable
from torch import Tensor

from torch import nn
from collections import OrderedDict

import torch.utils.data as data
import torch

import random
import numpy as np
import pandas as pd
from pathlib import Path
import math

from tqdm import tqdm

class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir: str,
                       input_file: str,
                       transforms: Optional[Callable]=None,
                       patch_size: int=256,
                       **kwargs) -> None:
        super(SliceDataset, self).__init__()
        
        self.split_set = kwargs.pop('split_set', 'train')
        
        self.input_dir = Path(input_dir)
        
        dfs = []
        for file in self.input_dir.iterdir():
            if file.is_file():
                if file.suffix == '.csv':
                    acquisition_id = file.stem.split('_')[-1]
                    df = pd.read_csv(file)
                    df.loc[:, 'acquisition_id'] = acquisition_id
                    df = df.set_index(['acquisition_id', 't', 'id'], drop=False)
                    if df.iloc[0].split_set == 'train':
                        dfs.append(df)

        df = pd.concat(dfs)
        self.df = df.sort_index()
        self.num_timesteps = len(self.df.index.unique(level='t'))
        
        self.transforms = transforms
        
        self.center_crop = kwargs.pop('center_crop', True)
        self.patch_size = patch_size
        self.dataset_size = kwargs.pop('dataset_size', 3200)
        self.time_sampling_bias = kwargs.pop('time_sampling_bias', 3.)
        self.dataset_name = kwargs.pop('dataset_name', '')
        
    def __getitem__(self, index: int) -> Mapping[str, Tensor]:
        s = random.random()
        a = self.time_sampling_bias
        t =  s*np.exp(-a*(1-s)) # sampling biased towards start of the trajectory
        
        timestep_index = int(t * (self.num_timesteps-1)) 
        
        row = self.df.xs(timestep_index, axis=0, level='t').sample().iloc[0]
        
        primal_step_size = row.primal_step_size
        lambda_reg = row.regularization_weight
        timestep = torch.tensor(row.t, dtype=torch.float32)
        
        joint_array = np.load(
            self.input_dir / row.joint_file, mmap_mode='r'
        )
        
        # height, width = input.shape; assert height == width
        height, width = joint_array.shape[1:]; assert height == width
        if height > 512:
            center_offset = height // 4
        else:
            center_offset = 50
        if self.center_crop:
            if height - self.patch_size - center_offset == center_offset:
                h_offset, w_offset = center_offset, center_offset
            else:
                h_offset = np.random.randint(center_offset, height - self.patch_size - center_offset)
                w_offset = np.random.randint(center_offset, width - self.patch_size - center_offset)
        elif self.patch_size == height:
            h_offset, w_offset = 0, 0
        else:
            h_offset = np.random.randint(height - self.patch_size)
            w_offset = np.random.randint(width - self.patch_size)
        
        joint_array = torch.from_numpy(joint_array[:, h_offset:h_offset+self.patch_size, w_offset:w_offset+self.patch_size].copy())

        outputs = OrderedDict()
        outputs['sparse_rc'], gradient_rc, outputs['reference_rc'] = joint_array.split(1, dim=0)

        if self.dataset_name == 'cork':
            L_max = 0.1179
        elif self.dataset_name == 'walnut':
            L_max = 0.502464
        
        outputs['sparse_rc'][:] = outputs['sparse_rc'] - primal_step_size * gradient_rc
        outputs['sparse_rc'][:] = outputs['sparse_rc'].clamp(0, None)

        return [value for value in outputs.values()] + [timestep]
    
    def __len__(self) -> int:
        return self.dataset_size     