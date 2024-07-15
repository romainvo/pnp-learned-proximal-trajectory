from typing import Callable, Any, Dict, Tuple, Union, Optional
import sys

import os
import time
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import cm
from PIL import Image

from timm.utils import ModelEmaV2

from utils import AverageMeter, count_parameters, update_summary, \
    save_checkpoint, Logger, parse_args
from model import create_model
from model.losses import *
from data import create_dataloader, create_transforms, create_dataset
from scheduler import create_scheduler, WarmUpWrapper
from optim import set_bn_weight_decay, create_optimizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

def main(args, args_text):

    global DEVICE 
    DEVICE = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu' 

    model = create_model(**vars(args))

    model_ema = None
    if args.ema:
        model_ema = ModelEmaV2(model, decay=args.ema_decay, device='cpu')

    checkpoint = None
    if args.checkpoint_file is not None:
        checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
        if args.iterative_model > 0:
            missing_keys, unexpected_keys = model._module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Missing keys :", missing_keys)
        print("Unexpected keys :", unexpected_keys)

        if 'state_dict_ema' in checkpoint:
            try:
                model_ema.load_state_dict(checkpoint['state_dict_ema'])
            except Exception as e:
                print("Iterative matching")
                for model_ema_v, ema_v in zip(model_ema.state_dict().values(), 
                                              checkpoint['state_dict_ema'].values()):
                    model_ema_v.copy_(ema_v)


    model = model.to(DEVICE)
    model = model.train()

    # Initialize the current working directory
    output_dir = ''
    output_base = args.output_base
    exp_name = '-'.join([
        datetime.now().strftime('%Y%m%d-%H%M%S'),
        'PostProcessing',
        type(model).__name__
    ])
    if args.output_suffix is not None:
        exp_name = '_'.join([exp_name, args.output_suffix])
    output_dir = os.path.join(output_base, args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train_imgs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val_imgs'), exist_ok=True)

    args.output_dir = output_dir

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(args_text)

    if args.log_name is not None:
        sys.stdout = Logger(os.path.join(output_dir, args.log_name+'.log'))
    else:
        print("****** NO LOG ******")

    print("Number of model parameters : {}\n".format(count_parameters((model))))

    print(args)

    train_dataset = create_dataset(transforms=create_transforms(training=args.augmentation),
                                   training=True,
                                   test=False,
                                   mode='postprocessing',
                                   outputs=['sparse_rc', 'reference_rc'],
                                   **vars(args))

    val_dataset = create_dataset(transforms=create_transforms(training=False),
                                 training=False,
                                 test=False,
                                 mode='postprocessing',
                                 outputs=['sparse_rc', 'reference_rc'],
                                 **vars(args))

    train_dataloader = create_dataloader(train_dataset, args.batch_size,
                                         num_workers=args.num_workers,
                                         trainval=True,
                                         shuffle=True,
                                         drop_last=args.drop_last,
                                         pin_memory=args.pin_memory)
    val_dataloader = create_dataloader(val_dataset, args.batch_size,
                                       num_workers=args.num_workers,
                                       trainval=True,
                                       shuffle=False,
                                       drop_last=args.drop_last,
                                       pin_memory=args.pin_memory)  

    loss_fn = create_loss(loss_name=args.loss)
    loss_fn = loss_fn.to(DEVICE)

    parameters = set_bn_weight_decay(model, weight_decay=0)
    optimizer = create_optimizer(parameters, **vars(args))

    last_epoch = -1
    args.num_step = len(train_dataloader) * args.num_epochs
    args.num_warmup_step = len(train_dataloader) * args.num_warmup_epochs
    lr_scheduler = create_scheduler(optimizer,
                                    name=args.lr_scheduler,
                                    num_epochs=args.num_epochs,
                                    num_epochs_restart=args.num_epochs_restart,
                                    lr_decay=args.lr_decay,
                                    patience=args.patience,
                                    threshold=args.plateau_threshold,
                                    milestones=args.milestones,
                                    min_lr=args.min_lr,
                                    last_epoch=last_epoch,
                                    )

    if args.num_warmup_epochs > 0:
        lr_scheduler = WarmUpWrapper(lr_scheduler, args.num_warmup_step, args.warmup_start)
        args.num_epochs += args.num_warmup_epochs

    if args.jacobian_spectral_norm_finetuning:
        print("****** JACOBIAN SPECTRAL NORM FINETUNING ******")
        print(f'****** {args.num_epochs} epochs ******')

    scaler = torch.cuda.amp.GradScaler(init_scale=2.0**15, enabled=args.amp) #64536 = 2**16

    best_metric = None
    best_epoch = None
    try:
        for epoch in range(args.num_epochs):
                train_metrics = train_epoch(
                    epoch, 
                    model, 
                    train_dataloader, 
                    optimizer, 
                    args, 
                    loss_fn, 
                    lr_scheduler, 
                    output_dir=args.output_dir, 
                    model_ema=model_ema,
                    scaler=scaler)

                if epoch % args.validation_interval == 0 or epoch == args.num_epochs - 1:
                    eval_metrics = validation(
                        model, 
                        val_dataloader, 
                        args, 
                        loss_fn, 
                        output_dir=args.output_dir,
                        epoch=epoch)

                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(args.output_dir, 'summary.csv'),
                    write_header=best_metric is None)  

                # Update the best_metric and best_epoch, only keep one checkpoint at a time
                if best_metric is None:
                    best_metric, best_epoch = eval_metrics[args.eval_metric], epoch
                    save_checkpoint(epoch, 
                                    model, 
                                    optimizer, 
                                    args,
                                    best_metric, 
                                    scheduler=lr_scheduler.state_dict(),
                                    state_dict_ema=model_ema.state_dict() if args.ema else None,
                                    ckpt_name=None)

                elif eval_metrics[args.eval_metric] < best_metric:
                    if os.path.exists(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch))):
                        os.unlink(os.path.join(args.output_dir, 'checkpoint-{}.pth.tar'.format(best_epoch)))

                    best_metric, best_epoch = eval_metrics[args.eval_metric], epoch
                    save_checkpoint(epoch, 
                                    model, 
                                    optimizer, 
                                    args, 
                                    best_metric, 
                                    scheduler=lr_scheduler.state_dict(),
                                    state_dict_ema=model_ema.state_dict() if args.ema else None,
                                    ckpt_name=None)

                save_checkpoint(epoch, 
                                model, 
                                optimizer, 
                                args, 
                                best_metric, 
                                scheduler=lr_scheduler.state_dict(),
                                state_dict_ema=model_ema.state_dict() if args.ema else None,
                                ckpt_name='last.pth.tar')

                if args.num_warmup_epochs > 0:
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        lr_scheduler.step(False, eval_metrics[args.eval_metric])
                    else:
                        lr_scheduler.step(False)                

                else:
                    if args.lr_scheduler == 'ReduceLROnPlateau':
                        lr_scheduler.step(eval_metrics[args.eval_metric])
                    else:
                        lr_scheduler.step()                
                

                print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

                if epoch > 3 and args.debug:
                    break

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1}) \n'.format(best_metric, best_epoch))
        save_checkpoint(epoch, 
                        model, 
                        optimizer, 
                        args, 
                        best_metric, 
                        state_dict_ema=model_ema.state_dict() if args.ema else None,
                        scheduler=lr_scheduler.state_dict(),
                        ckpt_name='last.pth.tar')

    if args.log_name is not None:
        sys.stdout.close()

    return best_metric

def train_epoch(epoch : int, 
                model : nn.Module, 
                loader : torch.utils.data.DataLoader, 
                optimizer : torch.optim.Optimizer, 
                args : Any, 
                loss_fn : Callable[[torch.Tensor], torch.Tensor], 
                lr_scheduler : Any, 
                output_dir : str='',
                model_ema: Optional[nn.Module]=None,
                scaler: Optional[torch.cuda.amp.GradScaler]=None,
                **kwargs) -> Dict[str, Union[float, int]]:

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    forward_time_m = AverageMeter()
    backward_time_m = AverageMeter()
    comm_time_m = AverageMeter()
    loss_m = AverageMeter()
    mse_m = AverageMeter()
    jacobian_spectral_norm_m = AverageMeter()

    model.train()
    end = time.time()

    last_idx = len(loader) -1
    num_step = len(loader) * epoch
    save_img = False
    for batch_idx, (batch) in enumerate(loader):
        data_time_m.update(time.time() - end)
        last_batch = last_idx == batch_idx
        
        timesteps = None
        noise_level = None
        if len(batch) == 2:
            input, target = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
        elif len(batch) == 3:
            input, target, cond = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            if args.indi:
                timesteps = cond.to(DEVICE, non_blocking=True)
            elif args.noise_level > 0.:
                noise_level = cond.to(DEVICE, non_blocking=True)

        comm_time_m.update(time.time() - end - data_time_m.val)

        forward_start = time.time()
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.amp):
            output = model(input, timesteps=timesteps,
                                  residual_learning=args.residual_learning,
                                  noise_level=noise_level)

            loss = loss_fn(output, target)

            jacobian_reg_loss = None           
            if args.jacobian_spectral_norm_finetuning and epoch > args.num_epochs - args.jacobian_spectral_norm_finetuning_epochs:
                jacobian_reg_loss, jacobian_avg_norm = jacobian_spectral_norm(model, 
                                                        input, 
                                                        target,
                                                        interpolation=args.pnp,
                                                        timesteps=timesteps,
                                                        num_power_steps=args.jacobian_spectral_norm_power_steps,
                                                        jacobian_reduce_mode='avg',
                                                        operator_mode='',
                                                        model_training=True,
                                                        residual_learning=False,
                                                        noise_level=noise_level) 
                jacobian_reg_loss = torch.clip(jacobian_reg_loss, 0)
                loss = loss + args.jacobian_spectral_norm_weight * jacobian_reg_loss

        forward_time_m.update(time.time() - forward_start)

        optimizer.zero_grad(set_to_none=False)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if np.random.rand() > 0.1 and not save_img:
            left = input[0].squeeze().detach().cpu().numpy().clip(0, None); left = left  / left.max() 
            right = output[0][0].squeeze().detach().cpu().numpy().clip(0,None); right = right / right.max()
            ckpt_img = np.zeros((left.shape[0], 2*left.shape[1]), dtype=np.float32)
            ckpt_img[:, :left.shape[1]] = left; ckpt_img[:, left.shape[1]:] = right; 
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(ckpt_img)*255))
            ckpt_img.save(os.path.join(output_dir, f'train_imgs/ckpt_img_{epoch}.png'))
            save_img = True

        if args.clip_grad_norm > 0 and args.clip_grad_value <= 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        elif args.clip_grad_value > 0 and args.clip_grad_norm <= 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
        elif args.clip_grad_value == 0 and args.clip_grad_norm == 0:
            pass
        else:
            raise ValueError("clip_grad_norm and clip_grad_value should be exclusives. Both cannot be > 0 at the same time", args.clip_grad_value, args.clip_grad_norm )

        scaler.step(optimizer)
        scaler.update()
        backward_time_m.update(time.time() - forward_start - forward_time_m.val)

        if args.ema:
            model_ema.update(model)

        with torch.no_grad():
            mse = F.mse_loss(output, target, reduction='mean')

        if jacobian_reg_loss is not None:
            jacobian_spectral_norm_m.update(jacobian_avg_norm.item(), input.size(0))    
        loss_m.update(loss.item(), input.size(0))
        mse_m.update(mse.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        if (batch_idx % args.log_interval == 0 and args.log_interval !=0) or last_batch:
            lr = optimizer.param_groups[0]['lr']
        
            print_string = (
                'Epoch {} - step [{}/{}] \n'
                'Epoch time: {batch_time.sum:.0f}s - {batch_time.val:.4f} s/step (avg: {batch_time.avg:.3f}) - {rate:.3f} img/s (avg: {rate_avg:.3f}) \n'
                'Forward time: {forward_time_rate:.2%} - {forward_time.val:.4f} s/step (avg: {forward_time.avg:.3f}) \n'
                'Backward time: {backward_time_rate:.2%} - {backward_time.val:.4f} s/step (avg: {backward_time.avg:.3f}) \n'
                'Data: {data_time_rate:.2%} - {data_time.val:.3f} s/step (avg: {data_time.avg:.3f}) - total : {data_time.sum:.3f}s \n'
                'Transfer time w/ gpu: {comm_time.val:.3f} s/step (avg: {comm_time.avg:.3f}) - total : {comm_time.sum:.3f}s \n'
                'LR: {lr:.3e} - Loss: {loss.val:.4e} (avg: {loss.avg:.2e}) - MSE : {mse.val:.4e} (avg: {mse.avg:.2e}) \n'.format(
                    epoch, batch_idx, last_idx,
                    batch_time=batch_time_m, rate=input.size(0) / batch_time_m.val, rate_avg=input.size(0) / batch_time_m.avg,
                    forward_time=forward_time_m, forward_time_rate=forward_time_m.avg / batch_time_m.avg,
                    backward_time=backward_time_m, backward_time_rate=backward_time_m.avg / batch_time_m.avg,
                    data_time=data_time_m, data_time_rate=data_time_m.avg / batch_time_m.avg,
                    comm_time=comm_time_m,
                    lr=lr, loss=loss_m, mse=mse_m)
            )
            if args.jacobian_spectral_norm_finetuning and epoch >= args.num_epochs - args.jacobian_spectral_norm_finetuning_epochs:
                print_string += f'              - Jacobian Spectral Norm: {jacobian_spectral_norm_m.val:.2e} (avg: {jacobian_spectral_norm_m.avg:.2e}) \n'

            print(print_string)

        if args.num_warmup_epochs > 0:
            lr_scheduler.step(warmup=True)   
        num_step += 1

        end = time.time()

        if args.debug and batch_idx > 5:
            break

    return_metrics = OrderedDict()
    return_metrics.update({'loss' : loss_m.avg})
    return_metrics.update({'mse' : mse_m.avg})
    return_metrics.update({'jacobian_spectral_norm' : jacobian_spectral_norm_m.avg})
    return_metrics.update({'lr' : lr})

    return return_metrics


def validation(model : nn.Module, 
               loader : torch.utils.data.DataLoader, 
               args : Any, 
               loss_fn : Callable[[torch.Tensor], torch.Tensor], 
               output_dir : str='',
               epoch: int=0) -> Dict[str, Union[float, int]]:

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    comm_time_m = AverageMeter()
    mse_m = AverageMeter()
    mae_m = AverageMeter()
    jacobian_reg_loss_m = AverageMeter()
    max_jacobian_spectral_norm = 0.

    model.eval()
    end = time.time()

    last_idx = len(loader) -1
    for batch_idx, (batch) in enumerate(loader):
        data_time_m.update(time.time() - end)
        last_batch = last_idx == batch_idx

        timesteps = None
        noise_level = None
        if len(batch) == 2:
            input, target = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
        elif len(batch) == 3:
            input, target, cond = batch
            input = input.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            if args.indi:
                timesteps = cond.to(DEVICE, non_blocking=True)
            elif args.noise_level > 0.:
                noise_level = cond.to(DEVICE, non_blocking=True)

        comm_time_m.update(time.time() - end)
        
        save_img = False
        jacobian_reg_loss, jacobian_norm_reduced = None, None
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.amp):
                output = model(input, timesteps=timesteps,
                                      residual_learning=args.residual_learning,
                                      noise_level=noise_level)

                mse = ((output - target) ** 2).mean()
                mae = (output - target).abs().mean()
                
                jacobian_reg_loss, jacobian_norm_reduced = jacobian_spectral_norm(model, 
                                                            input, 
                                                            target,
                                                            interpolation=False,
                                                            num_power_steps=args.jacobian_spectral_norm_power_steps,
                                                            operator_mode='',
                                                            eval_mode=True,
                                                            jacobian_reduce_mode='max',
                                                            timesteps=timesteps,
                                                            residual_learning=False, # We constrain the spectral norm of the model
                                                            noise_level=noise_level) # 

        if np.random.rand() > 0.1 and not save_img:
            left = input[0].squeeze().detach().cpu().numpy().clip(0, None); left = left  / left.max() 
            right = output[0][0].squeeze().detach().cpu().numpy().clip(0, None); right = right / right.max()
            ckpt_img = np.zeros((left.shape[0], 2*left.shape[1]), dtype=np.float32)
            ckpt_img[:, :left.shape[1]] = left; ckpt_img[:, left.shape[1]:] = right; 
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(ckpt_img)*255))
            ckpt_img.save(os.path.join(output_dir, f'val_imgs/ckpt_img_{epoch}.png'))
            save_img = True

        mse_m.update(mse.item(), input.size(0))
        mae_m.update(mae.item(), input.size(0))
        if jacobian_reg_loss is not None:
            max_jacobian_spectral_norm = max(max_jacobian_spectral_norm, jacobian_norm_reduced.item())
            jacobian_reg_loss_m.update(jacobian_reg_loss.item(), input.size(0))
        
        batch_time_m.update(time.time() - end)
        if last_batch:
        
            print(
                '[VALIDATION] Step [{}/{}] \n'
                'Elapsed time: {batch_time.sum:.0f}s - {batch_time.val:.4f} s/step (avg: {batch_time.avg:.3f}) - {rate:.3f} img/s (avg: {rate_avg:.3f}) \n'
                'MSE: {mse.val:.3e} (avg: {mse.avg:.2e}) \n'
                'MAE: {mae.val:.3e} (avg: {mae.avg:.2e}) \n'.format(
                    batch_idx, last_idx,
                    batch_time=batch_time_m, rate=input.size(0) / batch_time_m.val, rate_avg=input.size(0) / batch_time_m.avg,
                    mse=mse_m,
                    mae=mae_m)
            )  

        end = time.time()

        if args.debug and batch_idx > 5:
            break

    return_metrics = OrderedDict()
    return_metrics.update({'mse' : mse_m.avg})
    return_metrics.update({'mae' : mae_m.avg})
    return_metrics.update({'max_jacobian_spectral_norm' : max_jacobian_spectral_norm})
    return_metrics.update({'jacobian_reg_loss' : jacobian_reg_loss_m.avg})

    return return_metrics

if __name__ == '__main__':

    args, args_text = parse_args()
    main(args, args_text)
