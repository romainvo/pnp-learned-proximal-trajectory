import pandas as pd
import numpy as np
import os

import pathlib as path
from tqdm import tqdm

import tigre
import tigre_algs
import tigre.algorithms as tomo_algs
import time
from datetime import datetime
import math
import argparse
import gc
import torch

from tigre_algs.utils import SliceExtractor
from tigre.utilities.Ax import Ax
import skimage.transform as T

#### Test-time parser

def nullable_string(val):
    if not val:
        return None
    return val

def parse_args():

    parser = argparse.ArgumentParser(description = "Evaluate data-fidelity of set of reconstructions.")

    parser.add_argument('--input_dir', default='', type=str,
                        help='Path to the directory that contains the --input_file. All the paths referenced in the input file are relative to this path.')
    parser.add_argument('--input_file', default='dataset.csv', type=str,
                        help='Path to the input file containing the paths to the data')

    parser.add_argument('--dataset_name', default='cork', type=str, choices=['cork', 'walnut'])
    parser.add_argument('--split_set', default='test', type=str)
    parser.add_argument('--file_format', default='{acquisition_id}_pred.raw', type=str)
    parser.add_argument('--output_dir', default='', type=nullable_string)
    parser.add_argument('--output_suffix', default='', type=nullable_string)
    parser.add_argument('--bench_location', default='', type=nullable_string)
    
    parser.add_argument('--num_proj', default=50, type=int)


    args = parser.parse_args()
    return args

def create_geometry(args):
    
    input_dir = path.Path(args.input_dir)

    df = pd.read_csv(input_dir / args.input_file)
    df = df.loc[df.split_set == args.split_set]
    print(df)
    
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

    return geometry, ref, sinogram, angles, sparse_sinogram, sparse_angles

def main(args):

    input_dir = path.Path(args.input_dir)

    file_name = ''
    if args.output_suffix is not None:
        file_name += f'data_results_{args.split_set}_{args.output_suffix}.csv'
    else:
        file_name += f'data_results_{args.split_set}.csv'

    df = pd.read_csv(input_dir / args.input_file)
    df = df.loc[df.split_set == args.split_set]
    print(df)

    bench_dirs = {}
    if args.bench_location is not None:
        bench_loc = path.Path(args.bench_location)
        exp_dirs = []
        [exp_dirs.append(x) for x in bench_loc.iterdir() if x.is_dir()]
        for exp_dir in exp_dirs:
            bench_dirs[exp_dir.name.split('_')[-1]] = exp_dir

        output_dir = path.Path(args.bench_location)
    else:
        output_dir = path.Path(args.output_dir)

        for row in df.itertuples():
            bench_dirs[str(row.id)] = output_dir

    if args.dataset_name == 'cork':
        rc_shape = (df.iloc[0].detector_size, df.iloc[0].detector_size, df.iloc[0].detector_size)
    elif args.dataset_name == 'walnut':
        rc_shape = (df.iloc[0].number_of_slice, df.iloc[0].num_voxels, df.iloc[0].num_voxels)

    results = []

    for row_idx in tqdm(range(len(df)), total=len(df)):
        rc_file_name = ''
        row = df.iloc[row_idx]
        exp_dir = bench_dirs[str(row.id)]
        
        args.acquisition_id = row.id
        geometry, ref, sinogram, angles, sparse_sinogram, sparse_angles = create_geometry(args)

        rc_suffix = args.file_format.format(acquisition_id=row.id)
        if args.output_suffix is not None:
            rc_file_name += f'{rc_suffix}_{args.output_suffix}.raw'
        else:
            rc_file_name += f'{rc_suffix}.raw'
        if args.bench_location is not None:
            rc_file_name = f'{rc_suffix}_{args.output_suffix}.raw'

        output_rc = np.memmap(exp_dir / rc_file_name, 
                              dtype=np.float32, mode='r', 
                              shape=rc_shape)

        projected_rc = Ax(np.clip(output_rc,0,None), geometry, sparse_angles, 'interpolated')
        residual = (projected_rc - sparse_sinogram)**2
        fidelity_l2 = np.sum(residual)
        fidelity_mse = np.mean(residual)

        results.append({
            'id': row.id,
            'data_l2': fidelity_l2,
            'data_mse': fidelity_mse,
        })
        print(f'[results] {results[-1]}')

        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv(output_dir / file_name, index=False)          

    results.append({
        'id': 0,
        'data_l2': results_df.data_l2.mean(),
        'data_mse': results_df.data_mse.mean(),
    })

    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv(output_dir / file_name, index=False)               
    
if __name__ == '__main__':

    args = parse_args()

    main(args)
