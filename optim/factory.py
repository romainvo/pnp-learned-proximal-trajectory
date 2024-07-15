import torch
import torch.optim as optim

def create_optimizer(parameters, 
                     optimizer : str = 'adam',
                     init_lr : float=0.001,
                     momentum : float=0.,
                     dampening : float=0.,
                     nesterov : bool=False,
                     weight_decay : float=0.,
                     optimizer_eps : float=1e-8,
                     amsgrad : bool=False, 
                     **kwargs) -> optim.Optimizer:
    if optimizer == 'sgd':
        init_lr = kwargs.pop('init_lr', init_lr)
        optimizer = torch.optim.SGD(parameters,
                                    lr=init_lr,
                                    momentum=momentum,
                                    dampening=dampening,
                                    nesterov=nesterov,
                                    weight_decay=weight_decay)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr=init_lr,
                                     betas=(0.9, 0.999),
                                     eps=optimizer_eps,
                                     weight_decay=weight_decay,
                                     amsgrad=amsgrad)
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters,
                                      lr=init_lr,
                                      betas=(0.9, 0.999),
                                      eps=optimizer_eps,
                                      weight_decay=weight_decay,
                                      amsgrad=amsgrad)
    else:
        raise NotImplementedError()

    return optimizer