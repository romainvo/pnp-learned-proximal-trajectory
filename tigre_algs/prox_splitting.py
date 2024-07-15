from typing import Optional

import numpy as np
import copy
import pandas as pd
import os
import pathlib
from collections import OrderedDict
import gc

import tigre
from tigre.utilities.im_3d_denoise import im3ddenoise
from tigre.utilities.Atb import Atb
from tigre.utilities.Ax import Ax

from tigre.utilities.order_subsets import order_subsets

from tigre_algs import IterativeReconAlg, decorator
from .utils import *
from .deeptomo_utils import TorchModel

from tqdm import tqdm, trange
import time

from PIL import Image
from matplotlib import cm

import torch

class ProximalSplitting(IterativeReconAlg):
    __doc__ = (
        "Solves ConeBeam CT using proximal splitting method, either Proximal Gradient Descent or Chambolle-Pock algorithm\n"
    ) + IterativeReconAlg.__doc__

    def __init__(self, proj, geo, angles, niter, **kwargs):
        kwargs.update({"W": None, "V": None})
        self.blocksize = 20 if "blocksize" not in kwargs else kwargs["blocksize"]

        print("*****", geo.offOrigin.shape)

        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)
        self.res_prev = deepcopy(self.res)

        self.fidelity_mse = 0.
        self.relative_fidelity = 0.
        self.fidelity_gradient = 0.

        self.regularization_gradient = 0.
        self.relative_residual = 0.
        self.regularization_mse = 0.
        self.generalized_gradient = 0.
        self.prox_residual = 0.

        self.gradients = []
        self.summary_rows = []

        self.t = 1.0; self.t_prev = 1.0

        if self.pnp:
            self.postp_res = np.zeros_like(self.res, dtype=np.float32)
            
            self.denoising_model = TorchModel(self.reg_checkpoint, ema=self.ema)
            
            self.lambda_reg = kwargs.pop('lambda_reg', 1.)

            print(f'[init_lr]={self.init_lr}, [blocksize]={self.blocksize}, [lambda_reg]={self.lambda_reg}')
        else:
            print(f'[init_lr]={self.init_lr}, [blocksize]={self.blocksize}')
            
        if self.tv_denoising:
            print('********** TV DENOISING **********')
            print(f'[tv_lambda]={self.tvlambda}')
            self.dual_tv = np.zeros(tuple(self.geo.nVoxel) + (2,), dtype=np.float32) # we compute axial tv regularization

    def set_angle_index(self):
        """
        sets angle_index and angleblock if this is not given.
        :return: None
        """
        self.angleblocks, self.angle_index = order_subsets(
            self.angles, self.blocksize, self.OrderStrategy
        )

    def op_norm(self):
        k = 0
        
        geo, angle, k_iteration = self.prepare_geo(0)
        s_prev, s = 0, 0
        x = np.random.randn(*self.res.shape).astype(np.float32)
        b = np.zeros_like(self.proj, dtype=np.float32)

        while (abs(s_prev - s) > 1e-7 and k < 50) or s == 0:

            if self.tv_denoising:
                s_prev, x_prev = s, x
                b[:] = Ax(x_prev, geo, angle, "interpolated", gpuids=self.gpuids) 
                x[:] = Atb(b, geo, angle, "matched", gpuids=self.gpuids)
                x -= divergence_2d(gradient_2d(x_prev))    
                x[:] = x / np.linalg.norm(x)
                s = np.sqrt(
                        np.linalg.norm(Ax(x, geo, angle, "interpolated", gpuids=self.gpuids))**2
                      + np.linalg.norm(gradient_2d(x))**2)
            else:
                s_prev, x_prev = s, x
                b[:] = Ax(x_prev, geo, angle, "interpolated", gpuids=self.gpuids) 
                x[:] = Atb(b, geo, angle,"matched",gpuids=self.gpuids)
                x[:] = x / np.linalg.norm(x)
                s = np.linalg.norm(Ax(x, geo, angle, "interpolated", gpuids=self.gpuids))
                
            print(f'[{k}] s={s:.2e}', end=', ')
            k += 1
            
        return s

    def update_image(self, res_prev, geo, angle, k_iteration, iteration=0):
        ang_index = self.angle_index[k_iteration].astype(np.int)

        output = Ax(res_prev, geo, angle, "interpolated", gpuids=self.gpuids) # [num_angles, num_voxel_y, num_voxel_x]
        residual = (output - self.proj[ang_index]) # [num_angles, num_voxel_y, num_voxel_x]
        self.fidelity_mse = np.mean(residual**2)
        self.snr, self.psnr, self.mse = self.convergence_check()        

        self.gradient_step = (
            Atb(
                residual,
                geo,
                angle,
                "matched",
                gpuids=self.gpuids,
            )
        )

        self.fidelity_gradient = np.linalg.norm(self.gradient_step.reshape(-1), ord=2)
        self.relative_fidelity = np.linalg.norm(residual.reshape(-1), ord=2)**2 / np.linalg.norm(output.reshape(-1), ord=2)**2

        if self.slice_extractor is not None:
            self.slice_extractor.save_slices(
                t=iteration,
                current_iterate=res_prev,
                data_fidelity_gradient=self.gradient_step,
                target=self.dense_rc,
                primal_step_size=self.primal_lr,
                dual_step_size=1.0,
                regularization_weight=self.lambda_reg
            )

        self.res -= self.primal_lr  * self.gradient_step

    def minimize_primal_dual_gap(self, res_prev, iteration):
        geo, angle, k_iteration = self.prepare_geo(iteration)

        self.update_dual(res_prev, geo, angle, k_iteration)
        self.update_primal(res_prev, geo, angle, k_iteration, 
                           iteration=iteration)

    def update_primal(self, res_prev, geo, angle, k_iteration, iteration=0):
        ang_index = self.angle_index[k_iteration].astype(np.int)

        self.gradient_step = (
                Atb(
                    self.dual_res[ang_index],
                    geo,
                    angle,
                    "matched",
                    gpuids=self.gpuids,
                )
        )

        self.fidelity_gradient = np.linalg.norm(self.gradient_step.reshape(-1), ord=2)
        
        if self.slice_extractor is not None:
            self.slice_extractor.save_slices(
                t=iteration,
                current_iterate=res_prev,
                data_fidelity_gradient=self.gradient_step,
                target=self.dense_rc,
                primal_step_size=self.primal_lr,
                dual_step_size=self.dual_lr,
                regularization_weight=self.lambda_reg
            )
        
        self.res[:] = self.res - self.primal_lr * self.gradient_step 
        
        if self.tv_denoising:
            self.res += self.primal_lr * divergence_2d(self.dual_tv)
        
        self.res.clip(min=0, out=self.res)

    def update_dual(self, res_prev, geo, angle, k_iteration):
        if not hasattr(self, 'dual_res'):
            self.dual_res = np.zeros_like(self.proj, dtype=np.float32)

        ang_index = self.angle_index[k_iteration].astype(np.int)

        output = Ax(res_prev, geo, angle, "interpolated", gpuids=self.gpuids) # [num_angles, num_voxel_y, num_voxel_x]
        residual = (output - self.proj[ang_index]) # [num_angles, num_voxel_y, num_voxel_x]
        self.fidelity_mse = np.mean(residual**2)
        self.relative_fidelity = np.linalg.norm(residual.reshape(-1), ord=2)**2 / np.linalg.norm(output.reshape(-1), ord=2)**2
        
        self.snr, self.psnr, self.mse = self.convergence_check()

        self.dual_res[ang_index] = (self.dual_res[ang_index] + self.dual_lr * residual) / (1 + 1/self.dual_lr)
        
        if self.tv_denoising:
            self.dual_tv += self.dual_lr * gradient_2d(res_prev)
            dual_tv_norm = np.linalg.norm(self.dual_tv, axis=-1, keepdims=True)
            np.maximum(self.tvlambda, dual_tv_norm, out=dual_tv_norm)
            self.dual_tv /= (dual_tv_norm/self.tvlambda)
   
    def update_pnp(self, res_prev: np.ndarray, 
                          timesteps: Optional[int]=None):
        if not hasattr(self, 'first_postp'):
            self.first_postp = True

        batch_size = self.reg_batch_size

        if self.first_postp:          
            for k in trange(0, self.postp_res.shape[0], batch_size):
                batch_size = batch_size if k + batch_size <= self.postp_res.shape[0] else self.postp_res.shape[0] - k   
                inputs = OrderedDict({
                    'input': np.clip(res_prev[k:k+batch_size], 0, None)
                })
                if timesteps is not None:
                    inputs['timesteps'] = np.asarray(timesteps, dtype=np.float32)[None]
                else:
                    inputs['timesteps'] = None

                inputs['timesteps'] = np.asarray(0., dtype=np.float32)[None]
                self.postp_res[k:k+batch_size] = self.denoising_model(
                    inputs
                )
                
            self.first_postp = False

            output = self.postp_res[self.slice_index]
            res_slice = res_prev[self.slice_index]
            output = np.clip(output, 0, None)
            output_gt = np.clip(self.dense_rc[self.slice_index],0, None)

            if output.max() > 0:
                output = output / output.max()
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(output)*255))
            ckpt_img.save(self.output_dir / 'postp.png')  
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(output_gt / output_gt.max())*255))
            ckpt_img.save(self.output_dir / 'ground_truth.png')  
              
            prox_gamma = lambda gamma: (res_slice + gamma * output) / (1 + gamma)
            
            for gamma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                prox_out = prox_gamma(gamma)
                if prox_out.max() > 0:
                    prox_out = prox_out / prox_out.max()
                ckpt_img = Image.fromarray(np.uint8(cm.viridis(prox_out)*255))
                ckpt_img.save(self.output_dir / f'prox_{gamma}.png')          

        else:
           for k in trange(0, self.postp_res.shape[0], batch_size, leave=False):
                batch_size = batch_size if k + batch_size <= self.postp_res.shape[0] else self.postp_res.shape[0] - k   
                
                inputs = OrderedDict({
                    'input': np.clip(res_prev[k:k+batch_size], 0, None)
                })
                if timesteps is not None:
                    inputs['timesteps'] = np.asarray(timesteps, dtype=np.float32)[None]
                else:
                    inputs['timesteps'] = None

                inputs['timesteps'] = np.asarray(0., dtype=np.float32)[None]
                self.postp_res[k:k+batch_size] = self.denoising_model(
                    inputs
                ) 
        
        prox_step_size = self.primal_lr * self.lambda_reg
        denoising_prox = (res_prev + prox_step_size * self.postp_res) / (1 + prox_step_size)
        
        self.prox_residual = self.residual_norm(denoising_prox, self.res_prev)

        self.res[:] = denoising_prox + self.extrapolation_lr * (denoising_prox - res_prev)

    def update_ideal_denoiser(self, res_prev):
        if not hasattr(self, 'first_postp'):
            self.first_postp = True

        if self.first_postp:         
            self.first_postp = False

            output = self.dense_rc[self.slice_index]
            res_slice = res_prev[self.slice_index]
            output = np.clip(output, 0, None)

            if output.max() > 0:
                output = output / output.max()
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(output)*255))
            ckpt_img.save(self.output_dir / 'ground_truth.png')  
                     
            for gamma in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                prox_gamma = lambda gamma: (res_slice + gamma * output) / (1 + gamma)

                prox_out = prox_gamma(gamma)
                if prox_out.max() > 0:
                    prox_out = prox_out / prox_out.max()
                ckpt_img = Image.fromarray(np.uint8(cm.viridis(prox_out)*255))
                ckpt_img.save(self.output_dir / f'prox-gt_{gamma}.png')          

        prox_step_size = self.primal_lr * self.lambda_reg
        denoising_prox = (res_prev + prox_step_size * self.dense_rc) / (1 + prox_step_size)

        self.res[:] = denoising_prox + self.extrapolation_lr * (denoising_prox - res_prev)

    def prepare_geo(self, iteration):
        geo = copy.deepcopy(self.geo)

        k_iteration = iteration % len(self.angleblocks)
        if self.blocksize == 1:
            angle = np.array([self.angleblocks[k_iteration]], dtype=np.float32)
            angle_indices = np.array([self.angle_index[k_iteration]], dtype=np.int32)

        else:
            angle = self.angleblocks[k_iteration]
            angle_indices = self.angle_index[k_iteration]
            # slice parameters if needed
        geo.offOrigin = geo.offOrigin[angle_indices]
        geo.offDetector = geo.offDetector[angle_indices]
        geo.rotDetector = geo.rotDetector[angle_indices]
        geo.DSD = geo.DSD[angle_indices]
        geo.DSO = geo.DSO[angle_indices] 
        
        return geo, angle, k_iteration 

    def minimize_data_fidelity(self, res_prev, iteration):

        geo, angle, k_iteration = self.prepare_geo(iteration)

        self.update_image(res_prev, geo, angle, k_iteration, iteration=iteration)

    def acceleration_step(self):
        self.t = (1 + np.sqrt(1 + 4 * self.t_prev**2)) / 2
        relaxation_step = 1 + (self.t_prev - 1) / self.t
        
        self.res[:], self.res_prev[:] = self.res_prev + relaxation_step * (self.res - self.res_prev), self.res
        self.t_prev = self.t

    def run_main_iter(self):
        """
        Goes through the main iteration for the given configuration.
        :return: None
        """
        img_path = self.output_dir / f'imgs/{self.output_reconstruction_file}_k000.png'
        # output = self.res()[self.slice_index]
        output = self.res[self.slice_index]
        output = np.clip(output, 0, None)
        if output.max() > 0:
            output = output / output.max()
        ckpt_img = Image.fromarray(np.uint8(cm.viridis(output)*255))
        ckpt_img.save(img_path)  
        
        progress_bar = tqdm(range(self.niter), total=self.niter, leave=True, position=0)
        norm_x0 = self.norm(self.res)
        
        best_fidelity_epoch, best_fidelity = None, None
        best_mse_epoch, best_mse = None, None
        for i in progress_bar:   
            fw_start = time.time()
            if self.primal_dual:
                self.minimize_primal_dual_gap(self.res, i)
            elif self.fidelity:
                self.minimize_data_fidelity(self.res, i)
            else:
                self.snr, self.psnr, self.mse = self.convergence_check()        
            fw_end = time.time()
            
            img_path = self.output_dir / f'imgs/{self.output_reconstruction_file}_k{i-0.5}.png'
            output = self.res[self.slice_index]
            
            ## Saving log image
            output = np.clip(output, 0, None)
            if output.max() > 0:
                output = output / output.max()
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(output)*255))
            ckpt_img.save(img_path)  
            
            if self.ideal_denoiser:
                self.update_ideal_denoiser(self.res)
            if self.pnp:
                self.update_pnp(self.res, 
                                timesteps=i if self.timesteps else None)
            if self.noneg:
                self.res = self.res.clip(min=0)

            self.relative_residual = self.residual_norm(self.res, self.res_prev)
            self.generalized_gradient = np.linalg.norm((self.res - self.res_prev).reshape(-1), ord=2) / self.primal_lr
            
            if self.fista:
                self.acceleration_step()
            else:
                self.res_prev[:] = self.res

            img_path = self.output_dir / f'imgs/{self.output_reconstruction_file}_k{i}.png'
            output = self.res[self.slice_index]
            output = np.clip(output, 0, None)
            if output.max() > 0:
                output = output / output.max()
            ckpt_img = Image.fromarray(np.uint8(cm.viridis(output)*255))
            ckpt_img.save(img_path)  

            start = time.time()
            end = time.time()

            summary = {
                'step': i, 'train_epoch': i,
                'lr': self.init_lr,
                'norm_x0': norm_x0,
                'train_data_fidelity': self.fidelity_mse,
                'train_reg_loss': self.regularization_mse,
                'train_relative_residual': self.relative_residual,
                'train_relative_fidelity': self.relative_fidelity,
                'train_snr': self.snr,
                'train_psnr': self.psnr,
                'train_mse': self.mse,
                'train_fidelity_gradient': self.fidelity_gradient,
                'train_prox_residual': self.prox_residual,
                'train_regularization_gradient': self.regularization_gradient,
                'train_generalized_gradient': self.generalized_gradient,
            }
            self.summary_rows.append(summary)

            df = pd.DataFrame(self.summary_rows)
            df.to_csv(self.output_dir / 'summary.csv', index=False)

            description = f"[Data fidelity] - mse={self.fidelity_mse:.2e}. [PSNR]={self.psnr:.2f}dB. [SNR]={self.snr:.2f}dB. FW={fw_end-fw_start:.2f}s,"
            description += f" checktime={end-start:.2f}s."

            progress_bar.set_description(description)  

            if best_fidelity is None:
                best_fidelity, best_fidelity_epoch = self.fidelity_mse, i
                if self.ckpt_fidelity:
                    self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_best_fidelity.raw')
            elif self.fidelity_mse < best_fidelity:
                best_fidelity, best_fidelity_epoch = self.fidelity_mse, i
                if self.ckpt_fidelity:
                    self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_best_fidelity.raw')

            if best_mse is None:
                best_mse, best_mse_epoch = self.mse, i
                if self.ckpt_reconstruction:
                    self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_best_reconstruction.raw')
            elif self.mse < best_mse:
                best_mse, best_mse_epoch = self.mse, i
                if self.ckpt_reconstruction:
                    self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_best_reconstruction.raw')

            if i > 0 and self.ckpt_criterion:
                criterion = self.relative_residual / norm_x0
                if criterion <= 1e-4:
                    self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_criterion.raw')
                    self.ckpt_criterion = False
                    if self.stopping == 'criterion':
                        break

            if i % self.checkpoint_interval == 0 and self.checkpoint_interval > 0:
                self.res.tofile(self.output_dir / f'{self.output_reconstruction_file}_{i}.raw')  

proximal_splitting = decorator(ProximalSplitting, name="proximal_splitting")