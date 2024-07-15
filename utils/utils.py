from typing import Dict, Any, Union, Optional, Sequence, Mapping
import torch.nn as nn
import torch
import os
from collections import OrderedDict
import csv
import sys
import time

class TensorHook(object):
    """ Register a backward hook on a named Tensor. The TensorHook 
    holds the 'hold' of the hook object. The hook called on every 
    backward pass checks if any gradient is nan. 
    Using hooks is faster than setting torch.autograd.set_detect_anomaly(True)
    """
    def __init__(self, name : str, param : torch.Tensor) -> None:
        self.name = name
        self.hook = param.register_hook(self)
        
    def __call__(self, grad : torch.Tensor) -> None:
        grad_norm = grad[~torch.isnan(grad)].norm()
        assert torch.all(~torch.isnan(grad)), \
            "name={}, grad_shape={}, grad norm={}, num of nan={}".format(
                self.name, 
                grad.size(), 
                grad_norm, 
                torch.isnan(grad).sum()
        )
        
    def close(self):
        self.hook.remove()

class ModuleForwardHook(object):
    def __init__(self, name : str, module : torch.nn.Module, output_dir) -> None:
        self.name = name
        self.handle = module.register_forward_hook(self)
        self.output_dir = output_dir
    
    def __call__(self, input : torch.Tensor, output : torch.Tensor) -> None:
        pass

    def close(self):
        self.handle.remove()

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Timer:
    def __init__(self, timers : Sequence[str]):

        self.timers = {timer: AverageMeter() for timer in timers}
        self.reset()

    def update(self, timer : str, val : float):
        self.timers[timer].update(val)

    def start(self):
        self.ref = time.time()

    def reset(self):
        for timer in self.timers:
            self.timers[timer].reset()

    def __getattr__(self, __name):
        return self.timers[__name]

def save_checkpoint(epoch: int, 
                    model: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    args: Mapping[str, Any], 
                    metric: Union[float, int], 
                    ckpt_name : Optional[str]=None,
                    **kwargs) -> Dict[str, Any]:
    """ Dump the model and optimizer state_dicts, the ConfigArguments and the current best_metric into a Dict.
    Save and returns the Dict """
    save_state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
        'metric': metric,
    }

    for key in kwargs.keys():
        if kwargs[key] is not None:
            save_state[key] = kwargs[key]
    
    if ckpt_name is None:
        ckpt_name = 'checkpoint-{}.pth.tar'.format(epoch)
        
    torch.save(save_state, os.path.join(args.output_dir, ckpt_name))

    return save_state

def update_summary(epoch : int, 
                train_metrics : Dict[str, Union[float, int]], 
                eval_metrics : Optional[Dict[str, Union[float, int]]]=None, 
                filename : str='summary.csv', 
                write_header : bool=False) -> None:
    """ Save train and eval metrics for plotting and reviewing """
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics is not None:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  
            dw.writeheader()
        dw.writerow(rowd)

def count_parameters(model : nn.Module) -> int:
    """ Counts the number of learnable parameters of a given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close()
