import torch
import numpy as np

def create_scheduler(optimizer : torch.optim.Optimizer,
                     name : str = 'MultiStepLR',
                     lr_decay : float = 0.5,
                     verbose : bool = False,
                     min_lr: float=1e-6,
                     last_epoch: int=-1,
                     **kwargs) -> torch.optim.lr_scheduler._LRScheduler:

    if name == 'MultiStepLR':
        milestones = kwargs.pop('milestones', [20,80,150])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=lr_decay,
            verbose=verbose,
            last_epoch=last_epoch
        )

    elif name == 'CosineAnnealingLR':
        num_epochs = kwargs.pop('num_epochs', 300)
        num_epochs_restart = kwargs.pop('num_epochs_restart', -1)
        if num_epochs_restart > 0:
            T_max = num_epochs_restart
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=T_max,
                T_mult=1,
                eta_min=min_lr,
                last_epoch=last_epoch,
                verbose=verbose)
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=min_lr,
                last_epoch=last_epoch,
                verbose=verbose)

    elif name == 'ReduceLROnPlateau':
        patience = kwargs.pop('patience', 5)
        plateau_threshold = kwargs.pop('plateau_threshold', 0.005) # 0.5%
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_decay,
            patience=patience,
            threshold_mode='rel',
            threshold=plateau_threshold,
            cooldown=0,
            min_lr=min_lr,
            eps=1e-8,
            verbose=verbose,
            last_epoch=last_epoch
        )

    else:
        raise NotImplementedError

    return lr_scheduler