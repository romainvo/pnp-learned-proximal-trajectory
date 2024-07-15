import torch
import torch.nn as nn
import numpy as np
import random

import torchvision.transforms as transforms

from torch import Tensor
from typing import List, Dict

class Compose:
    def __init__(self, T):
        self.T = T

    def __call__(self, **array_dict):
        for transform in self.T:
            out = transform(array_dict)

        return out 

class OneOf:
    def __init__(self, T):
        self.T = T

    def __call__(self, batch_dict):        
        transform = np.random.choice(self.T)

        return transform(batch_dict) 

class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = 0.5

    def __call__(self, batch_dict):
        if random.random() < self.p:
            for key in batch_dict:
                batch_dict[key] = np.flip(batch_dict[key], axis=1)
        else:
            return batch_dict

class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = 0.5

    def __call__(self, batch_dict):
        if random.random() < self.p:
            for key in batch_dict:
                batch_dict[key] = np.flip(batch_dict[key], axis=0)
        else:
            return batch_dict

class AxisFlip:
    def __init__(self, p: float=0.5, axis: int=0):
        self.p = p
        self.axis = axis

    def __call__(self, batch_dict):
        if random.random() < self.p:
            for key in batch_dict:
                batch_dict[key] = np.copy(np.flip(batch_dict[key], axis=self.axis))
        else:
            return batch_dict

class ToTensor:
    def __init__(self, num_dimensions: int=2):
        self.num_dimensions = num_dimensions

    def __call__(self, batch_dict):
        if self.num_dimensions == 2:
            for key in batch_dict:
                if len(batch_dict[key].shape) < 3:
                    batch_dict[key] = torch.from_numpy(batch_dict[key].copy()).unsqueeze(0)
                else:
                    batch_dict[key] = torch.from_numpy(batch_dict[key].copy()).permute(2,0,1)

        elif self.num_dimensions == 3:
            for key in batch_dict:
                if len(batch_dict[key].shape) < 4:
                    batch_dict[key] = torch.from_numpy(batch_dict[key]).unsqueeze(0)
                else:
                    batch_dict[key] = torch.from_numpy(batch_dict[key]).permute(3,0,1,2)
                
        return batch_dict

class ComposeModule(nn.Module):
    def __init__(self, T : List[nn.Module]) -> None:
        super(ComposeModule, self).__init__()

        self.T = nn.Sequential(*T)

    @torch.no_grad()
    def forward(self, **batch_dict) ->  Dict[str, Tensor]:
        for transform in self.T:
            batch_dict = transform(batch_dict)

        return batch_dict 

class RandomVerticalFlipModule(nn.Module):
    def __init__(self, p=0.5):
        super(RandomVerticalFlipModule, self).__init__()
        self.p = p

    def forward(self, batch_dict : Dict[str, Tensor]) ->  Dict[str, Tensor]:
        if random.random() < self.p:
            for key in batch_dict:
                batch_dict[key] = transforms.functional.vflip(batch_dict[key])

        return batch_dict

class RandomHorizontalFlipModule(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipModule, self).__init__()
        self.p = p

    def forward(self, batch_dict : Dict[str, Tensor]) ->  Dict[str, Tensor]:
        if random.random() < self.p:
            for key in batch_dict:
                batch_dict[key] = transforms.functional.hflip(batch_dict[key])

        return batch_dict

def create_transforms(training : bool=True, module_like : bool=False):

    if training:
        if module_like:
            transform = ComposeModule([
                RandomVerticalFlipModule(p=0.5),
                RandomHorizontalFlipModule(p=0.5)
            ])
        else:
            transform = Compose([
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                ToTensor()
            ])
    else:
        if module_like:
            transform = ComposeModule([
                nn.Identity()
            ])
        else:
            transform = Compose([
                ToTensor()
            ])

    return transform