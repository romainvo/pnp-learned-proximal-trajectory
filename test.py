import sys
from pathlib import Path
import argparse
from tqdm import tqdm, trange

import torch
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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

#### Test-time parser

def nullable_string(val):
    if not val:
        return None
    return val

def parse_args():

    parser = argparse.ArgumentParser(description = "Deep Tomography test")

    parser.add_argument('--input_dir', default='', type=str,
                        help='Path to the directory that contains the --input_file. All the paths referenced in the input file are relative to this path.')
    parser.add_argument('--input_file', default='dataset.csv', type=str,
                        help='Path to the input file containing the paths to the data')

    parser.add_argument('--dataset_name', default='cork', type=str, choices=['cork', 'walnut'])
    parser.add_argument('--split_set', default='test', type=str)
    parser.add_argument('--file_format', default='{acquisition_id}_pred.raw', type=str)

    parser.add_argument('--output_suffix', default='', type=nullable_string)

    parser.add_argument('--bench_location', default='', type=nullable_string, 
                        help='Contain set of subdirs with the reconstruction files to be evluated in each subdir. It is exclusive with --bench_dir.')
    parser.add_argument('--bench_dir', default='', type=nullable_string, 
                        help='Used when only one reconstruction is to be evaluated. The reconstruction should be located at the root of <bench_dir>. It is exclusive with --bench_location.' )

    parser.add_argument('--memmap', action='store_true')
    parser.add_argument('--no-memmap', action='store_false', dest='memmap')

    parser.add_argument('--reg_checkpoint', default='', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--patch_size', default=256, type=int)
    parser.add_argument('--patch_stride', default=128, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_memory', action='store_true')

    parser.add_argument('--sparse_evaluation', action='store_true')

    args = parser.parse_args()
    return args

def main(args):

    bench_dirs = []
    if args.bench_location is not None:
        bench_loc = Path(args.bench_location)
        [bench_dirs.append(x) for x in bench_loc.iterdir() if x.is_dir()]

        output_dir = Path(args.bench_location)
    
    if args.bench_dir is not None:
        bench_dir = Path(args.bench_dir)
        bench_dirs.append(bench_dir)

        output_dir = Path(args.bench_dir)

    input_dir = Path(args.input_dir)
    df = pd.read_csv(input_dir / args.input_file)
    print(df.head())

    file_name = ''
    if args.output_suffix is not None:
        file_name += f'results_{args.split_set}_{args.output_suffix}.csv'
    else:
        file_name += f'results_{args.split_set}.csv'

    if args.dataset_name == 'cork':
        rc_shape = (df.iloc[0].detector_size, df.iloc[0].detector_size, df.iloc[0].detector_size)
        L_max = 0.1179

    elif args.dataset_name == 'walnut':
        rc_shape = (df.iloc[0].number_of_slice, df.iloc[0].num_voxels, df.iloc[0].num_voxels)

        L_max = 0.502464

    mse_m = AverageMeter()
    ssim_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_idep_m = AverageMeter()
    psnr_idep_m = AverageMeter()

    kernel = gaussian_kernel(11, sigma=1.5).view(1,1,-1).to(DEVICE)
    results = []

    for bench_dir in bench_dirs:

        # small hack to retrieve the acquisition_id from the file name
        rc_file_list = [elt.name for elt in bench_dir.iterdir() if (elt.is_file() and 'raw' in elt.name)]
        if len(rc_file_list) == 0:
            continue
        acquisition_id = rc_file_list[0].split('_')[0]

        acquisition_id = int(acquisition_id) if args.dataset_name == 'walnut' else acquisition_id

        row = df[df.id == acquisition_id].iloc[0]

        if args.memmap:
            reference_rc = np.memmap(input_dir / row.reconstruction_file, 
                                     dtype=np.float32, 
                                     mode='r', 
                                     shape=rc_shape)
            output_rc = np.memmap(bench_dir / args.file_format.format(acquisition_id=acquisition_id), 
                                  dtype=np.float32, 
                                  mode='r', 
                                  shape=rc_shape)

        else:
            reference_rc = np.fromfile(input_dir / row.reconstruction_file, 
                                       dtype=np.float32).reshape(rc_shape)
            output_rc = np.fromfile(bench_dir / args.file_format.format(acquisition_id=acquisition_id),
                                    dtype=np.float32).reshape(rc_shape)

        ssim_l = []
        ssim_l_idep = []
        psnr_l_idep = []

        with torch.no_grad():
            if args.dataset_name == 'cork':

                for slice_idx in trange(150, rc_shape[0] - 150, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((
                            np.clip(output_rc[150:-150, 256:256+512, 256:256+512], 0, None)
                            - np.clip(reference_rc[150:-150, 256:256+512, 256:256+512], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            elif args.dataset_name == 'walnut':
                for slice_idx in trange(100, rc_shape[0] - 100, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((
                            np.clip(output_rc[100:-100, 100:400, 100:400], 0, None)
                            - np.clip(reference_rc[100:-100, 100:400, 100:400], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            else:
                for slice_idx in trange(row.offset_top, row.offset_bottom, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx], device=DEVICE)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                mse = np.mean((
                            np.clip(output_rc[row.offset_top:row.offset_bottom], 0, None)
                            - np.clip(reference_rc[row.offset_top:row.offset_bottom], 0, None))**2)
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
        print(f'[results] {results[-1]}')

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
    print(f'[SSIM_IDEP] {ssim_idep_m.avg:.4f}')
    print(f'[PSNR_IDEP] {psnr_idep_m.avg:.2f}')

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

def sparse_evaluation(args):

    input_dir = Path(args.input_dir)
    output_dir = Path(args.input_dir)

    df = pd.read_csv(input_dir / args.input_file)
    df = df[df.split_set == args.split_set]
    print(df.head())

    if args.dataset_name == 'cork':
        rc_shape = (df.iloc[0].detector_size, df.iloc[0].detector_size, df.iloc[0].detector_size)
        L_max = 0.1179 

    elif args.dataset_name == 'walnut':
        rc_shape = (df.iloc[0].number_of_slice, df.iloc[0].num_voxels, df.iloc[0].num_voxels)

        L_max = 0.502464

    mse_m = AverageMeter()
    ssim_m = AverageMeter()
    psnr_m = AverageMeter()
    ssim_idep_m = AverageMeter()
    psnr_idep_m = AverageMeter()

    kernel = gaussian_kernel(11, sigma=1.5).view(1,1,-1).to(DEVICE)
    results = []

    for row in df.itertuples():
        acquisition_id = row.id

        if args.memmap:
            reference_rc = np.memmap(input_dir / row.reconstruction_file, 
                                     dtype=np.float32, 
                                     mode='r', 
                                     shape=rc_shape)
            output_rc = np.memmap(input_dir / row.sparse_reconstruction_file, 
                                  dtype=np.float32, 
                                  mode='r', 
                                  shape=rc_shape)

        else:
            reference_rc = np.fromfile(input_dir / row.reconstruction_file, 
                                       dtype=np.float32).reshape(rc_shape)
            output_rc = np.fromfile(input_dir / row.sparse_reconstruction_file,
                                    dtype=np.float32).reshape(rc_shape)

        ssim_l = []
        ssim_l_idep = []
        psnr_l_idep = []

        with torch.no_grad():
            if args.dataset_name == 'cork':

                for slice_idx in trange(150, rc_shape[0] - 150, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 256:256+512,256:256+512], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((
                            np.clip(output_rc[150:-150, 256:256+512, 256:256+512], 0, None)
                            - np.clip(reference_rc[150:-150, 256:256+512, 256:256+512], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            elif args.dataset_name == 'walnut':
                for slice_idx in trange(100, rc_shape[0] - 100, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx, 100:400, 100:400], device=DEVICE).clamp(min=0)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                    L_max_idep = (target.max() - target.min()).item()
                    ssim_idep, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max_idep)
                    ssim_l_idep.append(ssim_idep.item())

                    psnr_idep = 10 * np.log10(L_max_idep**2 / torch.mean((output - target)**2).item())
                    psnr_l_idep.append(psnr_idep)

                mse = np.mean((
                            np.clip(output_rc[100:-100, 100:400, 100:400], 0, None)
                            - np.clip(reference_rc[100:-100, 100:400, 100:400], 0, None))**2)
                psnr = 10 * np.log10(L_max**2 / mse)
                ssim = np.mean(ssim_l)

                ssim_idep = np.mean(ssim_l_idep)
                psnr_idep = np.mean(psnr_l_idep)

            else:
                for slice_idx in trange(row.offset_top, row.offset_bottom, desc='[SSIM]', position=1, leave=True):
                    target = torch.tensor(reference_rc[slice_idx], device=DEVICE).clamp(min=0)
                    output = torch.tensor(output_rc[slice_idx], device=DEVICE)

                    ssim, _ = piqa_ssim(target[None,None], output[None,None], kernel, value_range=L_max)
                    ssim_l.append(ssim.item())

                mse = np.mean((
                            np.clip(output_rc[row.offset_top:row.offset_bottom], 0, None)
                            - np.clip(reference_rc[row.offset_top:row.offset_bottom], 0, None))**2)
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
        print(f'[results] {results[-1]}')

        mse_m.update(mse, 1)
        ssim_m.update(ssim, 1)
        psnr_m.update(psnr.item(), 1)
        ssim_idep_m.update(ssim_idep, 1)
        psnr_idep_m.update(psnr_idep, 1)

    print(f'[EVALUTATION] {len(results)} samples with {len(output_rc)} slices each')
    print(f'[MSE] {mse_m.avg:.2e}')
    print(f'[PSNR] {psnr_m.avg:.2f}')
    print(f'[SSIM] {ssim_m.avg:.4f}')
    print(f'[SSIM_IDEP] {ssim_idep_m.avg:.4f}')
    print(f'[PSNR_IDEP] {psnr_idep_m.avg:.2f}')

    results.append({
        'id': 0,
        'mse': mse_m.avg,
        'psnr': psnr_m.avg,
        'ssim': ssim_m.avg,
        'ssim_idep': ssim_idep_m.avg,
        'psnr_idep': psnr_idep_m.avg
    })

    file_name = ''
    if args.output_suffix is not None:
        file_name += f'sparse_results_{args.split_set}_{args.output_suffix}.csv'
    else:
        file_name += f'sparse_results_{args.split_set}.csv'

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(output_dir / file_name, index=False) 

if __name__ == '__main__':

    args = parse_args()

    if args.sparse_evaluation:
        sparse_evaluation(args)
    else:
        main(args)
