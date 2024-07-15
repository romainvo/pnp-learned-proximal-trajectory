from torch import Tensor

import torch.nn as nn

def create_loss(loss_name : str) -> nn.Module:

    if loss_name == 'mse':
        loss_fn = nn.MSELoss(reduction='mean')
    elif loss_name == 'mae':
        loss_fn = nn.L1Loss(reduction='mean')
    else:
        raise NotImplementedError()

    return loss_fn