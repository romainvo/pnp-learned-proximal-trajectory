from typing import Callable, Any, Dict, Tuple, Union, Optional

from pathlib import Path
import time
from datetime import datetime
import argparse
from tqdm import tqdm, trange

import copy
from timm.utils import ModelEmaV2

import torch
import torchvision
import math
import numpy as np
import pandas as pd

from piqa.ssim import ssim as piqa_ssim
from piqa.ssim import gaussian_kernel

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

from model import create_model
from data.dataset.postprocessing import WalnutInferenceDataset, CorkInferenceDataset
from data import create_dataloader, create_transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

#### Test-time parser

def nullable_string(val):
    if not val:
        return None
    return val

parser = argparse.ArgumentParser(description = "Deep Tomography test")

parser.add_argument('--input_dir', default='', type=str,
                    help='Path to the directory that contains the --input_file. All the paths referenced in the input file are relative to this path.')
parser.add_argument('--checkpoint_file', default=None, type=str,
                    help='Path to the checkpoint file to resume training from.')
parser.add_argument('--input_file', default='dataset.csv', type=str,
                    help='Path to the input file containing the paths to the data')
parser.add_argument('--output_base', default='', type=str)
parser.add_argument('--output_dir', default='', type=nullable_string, 
                    help='path to output folder (default: none, will save directly at the root of output_base/)')
parser.add_argument('--output_suffix', default='', type=nullable_string)

parser.add_argument('--mode', default='inference', type=str, choices=['inference', 'test'])
parser.add_argument('--inference_save', action='store_true')

parser.add_argument('--dataset_name', default='cork', type=str)
parser.add_argument('--split_set', default='test', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--patch_size', default=256, type=int)
parser.add_argument('--patch_stride', default=256, type=int)

parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--pin_memory', action='store_true')
parser.add_argument('--amp', action='store_true')

if __name__ == '__main__':

    test_args = parser.parse_args()

    checkpoint_file = Path(test_args.checkpoint_file)
    output_dir = checkpoint_file.parent if test_args.output_dir is None else Path(test_args.output_dir)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    train_args = checkpoint['args']

    model = create_model(**vars(train_args))
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)

    if 'state_dict_ema' in checkpoint:
        try:
            print('*** New EMA loading ckpt ***')
            model.load_state_dict(checkpoint['state_dict_ema'], strict=True)
        except Exception as e:
            # print(e)
            print('*** Legacy EMA loading ckpt ***')
            model_ema = ModelEmaV2(model, decay=0.999, device='cpu')

            for model_v, ema_v in zip(model.state_dict().values(), 
                                        checkpoint['state_dict_ema'].values()):
                model_v.copy_(ema_v)

            # model = copy.deepcopy(model_ema.module)

    print("Missing keys :", missing_keys)
    assert len(missing_keys) == 0
    print("Unexpected keys :", unexpected_keys)

    model = model.eval().to(DEVICE)

    if test_args.dataset_name == 'cork':
        dataset = CorkInferenceDataset(
            input_dir=test_args.input_dir,
            input_file=test_args.input_file,
            patch_size=test_args.patch_size,
            patch_stride=test_args.patch_stride,
            transforms=create_transforms(training=False),
            final_activation=train_args.final_activation,
            outputs=['sparse_rc', 'reference_rc'],
            split_set=test_args.split_set,
            
        )
        L_max = 0.1179

    elif test_args.dataset_name == 'walnut':
        dataset = WalnutInferenceDataset(
            input_dir=test_args.input_dir,
            input_file=test_args.input_file,
            patch_size=test_args.patch_size,
            patch_stride=test_args.patch_stride,
            transforms=create_transforms(training=False),
            final_activation=train_args.final_activation,
            outputs=['sparse_rc', 'reference_rc'],
            split_set=test_args.split_set,
            
        )
        L_max = 0.502464

    dataloader = create_dataloader(
        dataset,
        batch_size=test_args.batch_size,
        num_workers=0,#train_args.num_workers,
        persistent_workers=False,
        shuffle=False,
        drop_last=False,
        pin_memory=test_args.pin_memory)

    mse_m = AverageMeter()
    ssim_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_idep_m = AverageMeter()
    psnr_idep_m = AverageMeter()

    patch_size = test_args.patch_size
    kernel = gaussian_kernel(11, sigma=1.5).view(1,1,-1).to(DEVICE)
    results = []

    file_name = ''
    if test_args.output_suffix is not None:
        file_name += f'results_{test_args.split_set}_{test_args.output_suffix}.csv'
    else:
        file_name += f'results_{test_args.split_set}.csv'

    outter_pbar = tqdm(total=len(dataset.df), desc='Test', position=0, leave=True)
    for row_idx in range(len(dataset.df)):
        row = dataloader.dataset.update_sample_list(row_idx)
        print(dataloader.dataset.sample_list)

        rc_file_name = ''
        if test_args.output_suffix is not None:
            rc_file_name += f'{row.id}_{test_args.output_suffix}_pred.raw'
        else:
            rc_file_name += f'{row.id}_pred.raw'

        if test_args.mode == 'test':
            output_rc = np.memmap(output_dir / rc_file_name, 
                                  dtype=np.float32, mode='r', 
                                  shape=(dataset.number_of_slice, dataset.num_voxels, dataset.num_voxels))

        elif test_args.mode == 'inference':

            output_rc = np.zeros((dataset.number_of_slice, dataset.num_voxels, dataset.num_voxels), dtype=np.float32)

            for batch_idx, (batch) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Batch', position=1, leave=True):

                input, target, row_id, slice_indexes, h_offsets, w_offsets = batch

                input = input.to(DEVICE)

                with torch.no_grad():
                    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=test_args.amp):
                        output = model(input, residual_learning=train_args.residual_learning)[:,0].clamp(min=0)

                        if patch_size > dataset.num_voxels:
                            output = torchvision.transforms.functional.center_crop(
                                output, 
                                output_size=(dataset.num_voxels, dataset.num_voxels))
                            # re-pad to patch_size, center the padding, take into account uneven images
                            padding_left = math.ceil((patch_size - dataset.num_voxels) / 2)
                            padding_right = patch_size - dataset.num_voxels - padding_left 

                for idx, (slice_idx, h_offset, w_offset) in enumerate(zip(slice_indexes, h_offsets, w_offsets)):
                    output_size = output[idx].shape[-1]
                    output_rc[slice_idx, 
                            h_offset:h_offset + output_size,
                            w_offset:w_offset + output_size] \
                    = output_rc[slice_idx, 
                            h_offset:h_offset + output_size,
                            w_offset:w_offset + output_size] \
                    + output[idx].cpu().squeeze().numpy().astype(np.float32)

            output_rc = output_rc / dataset.slice_counts[None]

        else:
            raise NotImplementedError()
        
        ssim_l = []
        ssim_l_idep = []
        psnr_l_idep = []

        with torch.no_grad():
            if test_args.dataset_name == 'cork':
                for slice_idx in trange(150, dataset.number_of_slice - 150, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(dataset.reference_rcs[row.id][slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((output_rc[150:-150, 256:256+512, 256:256+512] 
                            - np.clip(dataset.reference_rcs[row.id][150:-150, 256:256+512, 256:256+512], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            elif test_args.dataset_name == 'walnut':
                for slice_idx in trange(100, dataset.number_of_slice - 100, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(dataset.reference_rcs[row.id][slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((output_rc[100:-100, 100:400, 100:400] 
                            - np.clip(dataset.reference_rcs[row.id][100:-100, 100:400, 100:400], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            else:
                for slice_idx in trange(row.offset_top, row.offset_bottom, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(dataset.reference_rcs[row.id][slice_idx], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx], device=DEVICE)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                mse = np.mean((output_rc[row.offset_top:row.offset_bottom] 
                            - np.clip(dataset.reference_rcs[row.id][row.offset_top:row.offset_bottom], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

        results.append({
            'id': row.id,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'ssim_idep': ssim_idep,
            'psnr_idep': psnr_idep
        })

        if test_args.inference_save:
            output_rc.tofile(output_dir / rc_file_name)

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(output_dir / file_name, index=False)

        mse_m.update(mse, 1)
        ssim_m.update(ssim, 1)
        psnr_m.update(psnr.item(), 1)
        ssim_idep_m.update(ssim_idep, 1)
        psnr_idep_m.update(psnr_idep, 1)

    print(f'[EVALUTATION] {len(results)} samples with {len(output_rc)} slices each')
    print(f'[MSE] {mse_m.avg:.2e}')
    print(f'[PSNR] {psnr_m.avg:.2f}')
    print(f'[SSIM] {ssim_m.avg:.4f}')

    results.append({
        'id': 0,
        'mse': mse_m.avg,
        'psnr': psnr_m.avg,
        'ssim': ssim_m.avg,
        'ssim_idep': ssim_idep_m.avg,
        'psnr_idep': psnr_idep_m.avg
    })

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(output_dir / file_name, index=False)

