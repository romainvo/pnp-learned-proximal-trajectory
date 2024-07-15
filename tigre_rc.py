import pandas as pd
import numpy as np

import pathlib as path
from tqdm import tqdm

import tigre
import tigre_algs
import tigre.algorithms as tomo_algs
import time
from datetime import datetime
import argparse
import gc
import torch

from tigre_algs.utils import SliceExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="Classic reconstruction with TIGRE")

    parser.add_argument('--dataset_name', default='cork', type=str)

    parser.add_argument('--input_dir', default='/cuda/data-1/commun/Tomography/QualityCork3_AST', type=str)
    parser.add_argument('--input_file', default='dataset_50p_1v.csv', type=str)
    parser.add_argument('--output_base', default='tigre_reconstructions', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--output_suffix', default='', type=str)

    parser.add_argument('--no-fidelity', action='store_false', dest='fidelity')
    parser.add_argument('--num_proj', default=50, type=int)
    parser.add_argument('--num_full_proj', default=720, type=int)
    parser.add_argument('--block_size', default=50, type=int)
    parser.add_argument('--reconstruction_alg', default='', type=str)
    parser.add_argument('--init_mode', default='', type=str)
    parser.add_argument('--n_iter', default=500, type=int)
    parser.add_argument('--stopping', default='num_iter', type=str,
                        help='Stopping criterion. Either `num_iter` or `criterion`')
    parser.add_argument('--ckpt_fidelity', action='store_true')
    parser.add_argument('--ckpt_reconstruction', action='store_true')
    parser.add_argument('--ckpt_criterion', action='store_true')

    parser.add_argument('--init_lr', default=1e-5, type=float)
    parser.add_argument('--tv_w', default=0.004, type=float)

    parser.add_argument('--checkpoint_interval', default=1, type=int)

    parser.add_argument('--primal_dual', action='store_true')
    parser.add_argument('--ideal_denoiser', action='store_true')
    parser.add_argument('--no-ideal_denoiser', action='store_false', dest='ideal_denoiser')
    parser.add_argument('--pnp', action='store_true')
    parser.add_argument('--tv_denoising', action='store_true')
    
    parser.add_argument('--timesteps', action='store_true')
    parser.add_argument('--reg_checkpoint', default='', type=str)
    parser.add_argument('--ema', action='store_true', help='Use EMA weights if available')
    parser.add_argument('--num_gpus', default=1, type=int)

    parser.add_argument('--reg_batch_size', default=32, type=int)
    parser.add_argument('--lambda_reg', default=1., type=float)
    parser.add_argument('--extrapolation_lr', default=0., type=float)
    parser.add_argument('--fista', action='store_true')
    parser.add_argument('--no-fista', dest='fista', action='store_false')

    
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--save_trajectory', action='store_true')
    parser.add_argument('--trajectory_output_dir', default='')
    parser.add_argument('--trajectory_output_file', default='')

    parser.add_argument('--split_set', default='validation', type=str)
    parser.add_argument('--acquisition_id', default='', type=str)

    args = parser.parse_args()
    return args

def get_reconstruction_alg(name:str):

    reconstruction_alg = None

    if hasattr(tigre_algs, name):
        reconstruction_alg = getattr(tigre_algs, name)
    elif hasattr(tomo_algs, name):
        reconstruction_alg = getattr(tomo_algs, name)
    else:
        raise NotImplementedError()

    return reconstruction_alg


def main(args):

    try:
        input_dir = path.Path(args.input_dir)
        output_dir = path.Path(args.output_base) / args.output_dir

        exp_name = '-'.join([
            datetime.now().strftime('%Y%m%d-%H%M%S'),
            f'TIGRE_3DCT',
        ])
        if args.output_suffix:
            exp_name = f'{exp_name}_{args.output_suffix}'

        output_dir = output_dir / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)

        args.output_reconstruction_file = args.acquisition_id
        args.output_quality_file = f'{args.acquisition_id}_quality'

        df = pd.read_csv(input_dir / args.input_file)
        df = df.loc[df.split_set == args.split_set]
        print(df)
        
        init = args.init_mode
        offset_detector = [0,0]
        offset_origin = [0,0,0]
        roll = 0 
        if args.dataset_name == 'cork':
            row = df.loc[df.id == args.acquisition_id].iloc[0]

            num_voxels = row.detector_size
            num_full_proj = row.num_full_proj
            num_proj = args.num_proj

            angles = np.linspace(0, 2 * np.pi, num_full_proj, endpoint=False)
            # generate projections
            sparse_indexes = np.linspace(0, num_full_proj-1, num_proj, endpoint=True, dtype=int)

            detector_size = (204.8, 204.8) #mm
            n_detectors = (1024,1024)
            px_size = (200 * 1e-3, 200 * 1e-3)
            num_full_proj = 720
            source_to_obj = 190 #mm
            obj_to_det = 637 - 190 #mm
            
            # calculate geometrical magnification
            mag = 3.27 #(source_to_obj + obj_to_det) / source_to_obj
            voxel_size = min(detector_size) / (mag * num_voxels)

            sinogram = np.memmap(input_dir / row.sinogram_file, shape=(*n_detectors,num_full_proj),
                                dtype=np.float32, mode='c')

            ref = np.memmap(input_dir / row.reconstruction_file, shape=(num_voxels, num_voxels, num_voxels), dtype=np.float32, mode='c')

            sparse_sinogram = np.moveaxis(sinogram, -1, 0)[sparse_indexes]
            sparse_angles = angles[sparse_indexes][::-1]
                        
            if '.raw' in args.init_mode:
                init = np.memmap(input_dir / args.init_mode, shape=(num_voxels, num_voxels, num_voxels), dtype=np.float32, mode='c')

        elif args.dataset_name == 'walnut':
            row = df.loc[df.id == int(args.acquisition_id)].iloc[0]

            trajectory = np.load(input_dir / row.trajectory_file)

            num_voxels = row.num_voxels
            num_full_proj = row.num_full_proj
            num_proj = args.num_proj

            angles = np.linspace(0, 2 * np.pi, num_full_proj, endpoint=False)
            angles -= np.pi/2
            # generate projections
            sparse_indexes = np.linspace(0, num_full_proj-1, num_proj, endpoint=True, dtype=int)

            offset_origin = [0 - trajectory[0,2], 0, 0 - trajectory[0,0]]
            offset_detector = [trajectory[0,5] - trajectory[0,2], trajectory[0,3] - trajectory[0,0]] 
            roll = np.arcsin(trajectory[0,8] / 0.1496)

            detector_size = (145.41, 114.89) #mm
            n_detectors = (972, 768)
            num_full_proj = 1200
            px_size = (149.6e-3, 149.6e-3)
            # detector_size = detector_px * px_size
            source_to_obj = -trajectory[0,1] # 66 mm 
            obj_to_det = trajectory[0,4] # 133 mm
            source_to_det = source_to_obj + obj_to_det

            voxel_size = 100e-3
            mag = (source_to_obj + obj_to_det) / source_to_obj 

            sinogram = np.memmap(input_dir / row.sinogram_file, 
                                 shape=(num_full_proj,row.sinogram_height,row.sinogram_width),
                                 dtype=np.float32, mode='r')

            ref = np.memmap(input_dir / row.reconstruction_file, shape=(num_voxels, num_voxels, num_voxels), dtype=np.float32, mode='r')

            sparse_sinogram = np.clip(sinogram[sparse_indexes],0, None)
            sparse_angles = angles[sparse_indexes]
            
            if '.raw' in args.init_mode:
                init = np.memmap(input_dir / args.init_mode, shape=(num_voxels, num_voxels, num_voxels), dtype=np.float32, mode='c')

        geometry = tigre.geometry()
        geometry.DSD = float(source_to_obj + obj_to_det)
        geometry.DSO = float(source_to_obj)

        geometry.nDetector = np.array(n_detectors) 
        geometry.dDetector = np.array(px_size)
        geometry.sDetector = geometry.nDetector * geometry.dDetector #np.array(detector_size)

        geometry.nVoxel = np.array([num_voxels, num_voxels, num_voxels])
        geometry.dVoxel = np.array([voxel_size, voxel_size, voxel_size]) # np.array([object_size, object_size, object_size])
        geometry.sVoxel = geometry.nVoxel * geometry.dVoxel # np.array([object_size, object_size, object_size])

        geometry.offOrigin = np.array(offset_origin)  # Offset of image from origin   (mm)
        geometry.offDetector = np.array(offset_detector) 

        # Variable to define accuracy of 'interpolated' projection. It defines the amoutn of samples per voxel.
        # Recommended <=0.5    (vx/sample)
        geometry.accuracy = 0.5

        # y direction displacement for centre of rotation correction (mm).
        # This can also be defined per angle
        geometry.COR = 0.

        # Rotation of the detector, by X,Y and Z axis respectively. (rad)
        # This can also be defined per angle
        geometry.rotDetector = np.array([roll, 0, 0])

        geometry.mode = "cone"  # Or 'parallel'. Geometry type.

        verbose = True
        qualmeas = ["RMSE"]
        if 'pytigre-2.2.0' in tigre.__dict__['__path__'][0]:
            order = "random2"
        else:
            order = "random" # "random2" for pytigre >= 2.2 
        # random -> blocks are made of neighbouring projections
        # random2 -> blocks are made of random projections

        start = time.time()
        # quality_... = [l2_error, qualmeas]
        print("***** *******", geometry.offOrigin.shape)

        slice_extractor = None
        if args.save_trajectory:
            slice_extractor = SliceExtractor(
                output_dir=args.trajectory_output_dir,
                output_file=args.trajectory_output_file,
                acquisition_id=args.acquisition_id,
                num_steps=args.n_iter,
                num_proj=args.num_proj,
                num_full_proj=args.num_full_proj,
                split_set=args.split_set,
            )

        quality = None
        if args.reconstruction_alg == 'fdk':
            rc = tomo_algs.fdk(
                sparse_sinogram,
                geometry,
                sparse_angles,
                filter='shepp_logan',
            )
        else:
            reconstruction_alg = get_reconstruction_alg(args.reconstruction_alg)

            rc, quality = reconstruction_alg(
                sparse_sinogram, 
                geometry, 
                sparse_angles, 
                args.n_iter,
                verbose=verbose,
                Quameasopts=qualmeas,
                OrderStrategy=order,
                computel2=False,
                blocksize=args.block_size,
                tvlambda=args.tv_w,
                init=init,
                #####
                checkpoint_interval=args.checkpoint_interval,
                output_dir=output_dir,
                output_reconstruction_file=args.output_reconstruction_file,
                dense_rc=ref,
                #####
                fidelity=args.fidelity,
                primal_dual=args.primal_dual,
                ideal_denoiser=args.ideal_denoiser,
                pnp=args.pnp,
                tv_denoising=args.tv_denoising,
                slice_extractor=slice_extractor,
                timesteps=args.timesteps,
                #####
                reg_checkpoint=args.reg_checkpoint,
                ema=args.ema,
                reg_batch_size=args.reg_batch_size,
                lambda_reg=args.lambda_reg,
                init_lr=args.init_lr,
                num_gpus=args.num_gpus,
                extrapolation_lr=args.extrapolation_lr,
                fista=args.fista,
                stopping=args.stopping,
                ckpt_fidelity=args.ckpt_fidelity,
                ckpt_reconstruction=args.ckpt_reconstruction,
                ckpt_criterion=args.ckpt_criterion,
                #####
                dataset_name=args.dataset_name,
                num_proj=args.num_proj,
            )

    except KeyboardInterrupt:
        pass

    print(f'{(time.time()-start)/60:.2f} mn')
    rc.tofile(output_dir / f'{args.output_reconstruction_file}_{args.n_iter-1}.raw')   

if __name__ == '__main__':

    args = parse_args()
    input_dir = path.Path(args.input_dir)
    
    if args.save_trajectory:
        for split_set in ['train', 'validation']:
            df = pd.read_csv(input_dir / args.input_file)
            df = df.loc[df.split_set == split_set]
            for row in df.itertuples():
                print("Reconstructing", row.id, "...")
                args.acquisition_id = row.id
                args.split_set = row.split_set
                args.output_suffix = f'ideal_lfbs_fista_lambda{args.lambda_reg}_{args.num_proj}p_{row.id}'
                args.trajectory_output_file = f'ideal_lfbs_{args.num_proj}p_{row.id}'
                args.split_set = split_set

                main(args)
    elif args.benchmark:
        df = pd.read_csv(input_dir / args.input_file)
        df = df.loc[df.split_set == args.split_set]
        template_suffix = args.output_suffix
        for row in df.itertuples():
            print("Reconstructing", row.id, "...")
            args.acquisition_id = row.id
            args.split_set = row.split_set
            args.output_suffix = f'{template_suffix}_{row.id}'

            main(args) 
            gc.collect()
            torch.cuda.empty_cache()
              
    else:
        main(args)