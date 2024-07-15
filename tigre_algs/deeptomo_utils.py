from typing import Union, Mapping
from pathlib import Path

import torch
import numpy as np
import math

from model import create_model as create_postp_model
from timm.utils import ModelEmaV2

class TorchModel():
    def __init__(self, checkpoint_file: Union[str, Path],
                       ema: bool=True,
                       device: str='cuda:1'):
        
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        train_args = checkpoint['args']
        
        self.model = create_postp_model(**vars(train_args))
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        if 'state_dict_ema' in checkpoint and ema:
            try:
                print('*** New EMA loading ckpt ***')
                self.model.load_state_dict(checkpoint['state_dict_ema'], strict=True)

            except Exception as e:
                # print(e)
                print('*** Legacy EMA loading ckpt ***')

                for model_v, ema_v in zip(self.model.state_dict().values(), 
                                          checkpoint['state_dict_ema'].values()):
                    model_v.copy_(ema_v)
                    
        self.device = torch.device(device)
        print(self.device, torch.cuda.is_available())
        self.model.eval()
        self.model.to(self.device, dtype=None, non_blocking=True)
        self.residual_learning = train_args.residual_learning
        self.noise_level = bool(train_args.noise_level)
        print('Loading model.. Done.')
        
    def __call__(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        num_voxels = inputs['input'].shape[-1]
        
        # pad the input to next factor of 32 (for compatibility with model stride)
        pad = (32 - num_voxels % 32) % 32
        padding_left = math.ceil(pad / 2)
        padding_right = pad - padding_left        
        
        with torch.no_grad():
            for input_name in inputs:
                if inputs[input_name] is None:
                    continue
                if len(inputs[input_name].shape) == 2:
                    inputs[input_name] = inputs[input_name][None,None]
                elif len(inputs[input_name].shape) == 3:
                    inputs[input_name] = inputs[input_name][:,None]
                
                inputs[input_name] = torch.from_numpy(inputs[input_name]).to(self.device, dtype=torch.float16, non_blocking=True)
                
            noise_level = None
            if self.noise_level:
                noise_level = torch.tensor([0.001], device=self.device, dtype=torch.float16).repeat(inputs['input'].shape[0])
                                
            if pad > 0:
                for input_name in inputs:
                    if 'time' in input_name or inputs[input_name] is None:
                        continue
                    inputs[input_name] = torch.nn.functional.pad(inputs[input_name], (padding_left, padding_right, padding_left, padding_right), mode='constant', value=0)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = self.model(inputs['input'],
                                     timesteps=inputs['timesteps'],     
                                     noise_level=noise_level,
                                     residual_learning=self.residual_learning).squeeze()
        
            if pad > 0:
                outputs = outputs[...,
                                  padding_left:-padding_right,
                                  padding_left:-padding_right]
        
        return outputs.cpu().numpy().astype(np.float32)