from typing import Callable, Optional
from torch import Tensor

import torch
from torch import nn

def power_iteration(operator: Callable[[Tensor], Tensor], 
                    device: torch.device,
                    vector_size: int, 
                    steps: int=50,
                    momentum=0.0, 
                    eps=1e-3,
                    init_vec=None, 
                    verbose=False):
    '''
    Power iteration algorithm for spectral norm calculation
    '''
    with torch.no_grad():
        if init_vec is None:
            vec = torch.rand(vector_size).to(device)
        else:
            vec = init_vec.to(device)
        vec /= torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1) # shape: (batch_size,c,h,w)

        for i in range(steps):

            new_vec = operator(vec)
            new_vec = new_vec / torch.norm(new_vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1, 1)
            if momentum>0 and i > 1:
                new_vec -= momentum * old_vec
            old_vec = vec
            vec = new_vec
            
            diff_vec = torch.norm(new_vec - old_vec,p=2)
            if diff_vec < eps:
                if verbose:
                    print("Power iteration converged at iteration: ", i)
                break

    new_vec = operator(vec)
    div = torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0]) # shape: (batch_size,)
    lambda_estimate = torch.abs(
            torch.sum(vec.view(vector_size[0], -1) * new_vec.view(vector_size[0], -1), dim=1)) / div # shape: (batch_size,)

    return lambda_estimate

def jacobian_spectral_norm(
                       model: nn.Module,
                       input: Tensor,
                       target: Optional[Tensor]=None,
                       interpolation: bool=False,
                       num_power_steps: int=5,
                       epsilon: float=5e-2,
                       operator_mode: str='double_backward',
                       eval_mode: bool=False,
                       model_training: bool=False,
                       jacobian_reduce_mode: str='max',
                       **kwargs) -> Tensor:
        
        '''
        Compute the spectral norm of the model jacobian between at input point (or interpolation between input and target) -> largely taken from https://github.com/samuro95/Prox-PnP/blob/main/GS_denoising/lightning_denoiser.py
        
        '''
        
        model_mode = model.training
        
        with torch.enable_grad():
        
            if interpolation:
                eta = torch.rand(target.size(0), 1, 1, 1, requires_grad=True).to(target.device)
                x = eta * input.detach() + (1 - eta) * target.detach()
                x = x.to(target.device)
            else:
                x = input
            
            x.requires_grad_(True)
            
            model.train(model_training)
            
            x_hat = model(x, **kwargs)
            
            if operator_mode == 'double_backward':
                # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
                def operator(vec):
                    w = torch.ones_like(x_hat, requires_grad=True)
                    return torch.autograd.grad(torch.autograd.grad(x_hat, x, w, create_graph=True), w, vec, create_graph=not eval_mode)[0]  # Ju
            else:
                operator = lambda vec: torch.autograd.grad(x_hat, x, grad_outputs=vec, create_graph=not eval_mode, retain_graph=True, only_inputs=True)[0]

            jacobian_norm = power_iteration(operator, 
                                            x.device,
                                            x.size(), 
                                            steps=num_power_steps)

            jacobian_loss = torch.maximum(jacobian_norm, 
                                          torch.ones_like(jacobian_norm) - epsilon).mean()
            
            if jacobian_reduce_mode == 'max':
                jacobian_norm_reduced = torch.max(jacobian_norm).detach()
            else:
                jacobian_norm_reduced = torch.mean(jacobian_norm).detach()
            
            model.train(mode=model_mode)

            return jacobian_loss, jacobian_norm_reduced