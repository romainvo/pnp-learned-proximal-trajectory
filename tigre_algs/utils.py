from typing import Tuple, Union, Mapping, Optional
import numpy as np

import pandas as pd
from pathlib import Path

from copy import deepcopy

class SliceExtractor:
    def __init__(self,
                 output_dir: str,
                 output_file: str,
                 acquisition_id: str,
                 num_slice_per_reconstruction: int=2048,
                 num_steps: int=20,
                 num_proj: int=50,
                 split_set: str='train',
                 num_full_proj: int=720) -> None:
                
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'joint').mkdir(exist_ok=True)

        self.output_file = self.output_dir / f'{output_file}.csv'
        self.acquisition_id = acquisition_id
        self.num_slice_per_reconstruction = num_slice_per_reconstruction
        self.num_steps = num_steps
        self.num_proj = num_proj
        self.split_set = split_set
        
        # self.num_slice_per_step = int(self.num_slice_per_reconstruction / self.num_steps)
        self.num_slice_per_step = 20
        
        self.template_row = {
            'id': 0,
            'acquisition_id': self.acquisition_id,
            'num_full_proj': num_full_proj, 
            'num_proj': num_proj,
            'split_set': split_set,
            'slice_index': 0,
            'joint_file': 'joint/{}',
            'reconstruction_file': 'reconstruction/{}',
            'sparse_reconstruction_file': 'sparse_reconstruction_{}/{}',
            'sparse_gradient_file': 'sparse_gradient_{}/{}',
            't': 0,
            'primal_step_size': 1,
            'dual_step_size': 1,
            'regularization_weight': 1
        }
        
        self.df = None
        self.current_id = 0 
        
    def save_slices(self,
                    t: float,
                    current_iterate: np.ndarray,
                    data_fidelity_gradient: np.ndarray,
                    target: np.ndarray,
                    primal_step_size: float,
                    dual_step_size: float,
                    regularization_weight: float) -> None:
        
        # current_iteration : [num_voxels, num_voxels, num_voxels]
        # data_fidelity_gradient: [num_voxels, num_voxels, num_voxels]
        # target: [num_voxels, num_voxels, num_voxels]
        
        output_rows = []
        
        # we sample with axial bias (far from the top/bottom edge to avoid cone-beam artifacts)
        z_size = current_iterate.shape[0]
        sampled_slices = np.random.choice(np.arange(int(0.2*z_size), 
                                                    int(0.8*z_size),
                                                    step=1,
                                                    dtype=np.int32),
                                          size=self.num_slice_per_step,
                                          replace=False)
        
        for k, slice_index in enumerate(sampled_slices):
            output_rows.append(deepcopy(self.template_row))
            output_rows[k]['id'] = self.current_id
            output_rows[k]['slice_index'] = slice_index
            output_rows[k]['t'] = t
            output_rows[k]['primal_step_size'] = primal_step_size
            output_rows[k]['dual_step_size'] = dual_step_size
            output_rows[k]['regularization_weight'] = regularization_weight
            
            input_sample = current_iterate[slice_index]
            gradient_sample = data_fidelity_gradient[slice_index]
            target_sample = target[slice_index]
            
            file_name = f'{self.acquisition_id}/{t}/{self.acquisition_id}_{self.current_id}_t{output_rows[k]["t"]}.npy'
            
            joint_array = np.stack([input_sample, gradient_sample, target_sample], axis=0)
            output_rows[k]['joint_file'] = output_rows[k]['joint_file'].format(file_name)
            (self.output_dir / output_rows[k]['joint_file']).parent.mkdir(exist_ok=True, parents=True)
            np.save(self.output_dir / output_rows[k]['joint_file'], joint_array)
            
            output_rows[k]['reconstruction_file'] = output_rows[k]['reconstruction_file'].format(file_name)
            output_rows[k]['sparse_gradient_file'] = output_rows[k]['sparse_gradient_file'].format(output_rows[k]['num_proj'], file_name)
            output_rows[k]['sparse_reconstruction_file'] = output_rows[k]['sparse_reconstruction_file'].format(output_rows[k]['num_proj'], file_name)
            
            self.current_id += 1
            
        if self.df is None:
            self.df = pd.DataFrame.from_dict(output_rows)
        else:
            self.df = pd.concat(
                [self.df, pd.DataFrame.from_dict(output_rows)],
            )
            
        self.df.to_csv(self.output_dir / self.output_file, index=False) 

def gradient_2d(image: np.ndarray, out: Optional[np.ndarray]=None) ->  np.ndarray:
    """
    Compute the 2d spatial gradient of an n-dimensional array.
    """
    
    if out is None:
        out = np.zeros(image.shape + (2,), dtype=image.dtype)
    
    out[..., 0] = np.roll(image, -1, axis=-2) - image
    out[..., 1] = np.roll(image, -1, axis=-1) - image
    
    return out

def divergence_2d(image: np.ndarray, out: Optional[np.ndarray]=None) -> np.ndarray:
    """
    Compute the 2d spatial divergence of an n-dimensional array.
    """
    
    if out is None:
        out = np.zeros(image.shape[:-1], dtype=image.dtype)
    
    out += image[..., 0] - np.roll(image[..., 0], 1, axis=-2)
    out += image[..., 1] -  np.roll(image[..., 1], 1, axis=-1)
    
    return out
    
    
    