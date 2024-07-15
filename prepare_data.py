#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import astra
import os
import imageio
import time
import matplotlib.pyplot as plt

from zipfile import ZipFile
import shutil

import pathlib as path
import tifffile as tiff

# set the path to the data
input_dir = path.Path('data/Walnut-CBCT')

dense_reconstruction_dir = 'dense_reconstruction'; os.makedirs(input_dir / dense_reconstruction_dir, exist_ok=True)
sinogram_dir = 'sinogram'; os.makedirs(input_dir / sinogram_dir, exist_ok=True)
sparse_reconstruction_dir = 'sparse_reconstruction_50'; os.makedirs(input_dir / sparse_reconstruction_dir, exist_ok=True)

ref_name_rc = 'Walnut{}_GT.raw'
sparse_name_rc = 'Walnut{}_pos{}_FDK.raw' # 'fdk_pos{}_ass{}_vmm{}_{:06}.tiff'.format(orbit_id, angluar_sub_sampling, voxel_per_mm, i)
sinogram_name = 'Walnut{}_pos{}.raw'

for walnut_id in range(1,43):

    #### user defined settings #####################################################

    # select the ID of the sample you want to reconstruct
    # walnut_id = 1
    # select also the orbit you want to reconstruct the data from:
    # 1 higher source position, 2 middle source position, 3 lower source position
    orbit_id = 2
    # define a sub-sampling factor in angular direction
    # (all reference reconstructions are computed with full angular resolution)
    num_full_proj = 1200
    num_proj = 50

    sparse_indexes = np.linspace(0, 1200, num=num_proj, dtype=int, endpoint=False)
    angular_subsampling = 24 #24
    # select of voxels per mm in one direction (higher = larger res)
    # (all reference reconstructions are computed with 10)
    voxel_per_mm = 10

    # we enter here some intrinsic details of the dataset needed for our reconstruction scripts
    # set the variable "data_path" to the path where the dataset is stored on your own workstation
    exp_dir = input_dir / f'Walnut{walnut_id}'

    with ZipFile(input_dir / f'Walnut{walnut_id}.zip', 'r') as file_in:
        file_in.extractall(input_dir)

    #### load data #################################################################

    t = time.time();
    print('load data', flush=True)

    # we add the info about walnut and orbit ID
    data_path_full = exp_dir / 'Projections/tubeV{}'.format(orbit_id)
    projs_name = 'scan_{:06}.tif'
    dark_name = 'di000000.tif'
    flat_name = ['io000000.tif', 'io000001.tif']
    vecs_name = 'scan_geom_corrected.geom'
    projs_rows = 972
    projs_cols = 768

    # load the numpy array describing the scan geometry from file
    vecs = np.loadtxt(os.path.join(data_path_full, vecs_name))
    # get the positions we need; there are in fact 1201, but the last and first one come from the same angle
    vecs       = vecs[range(0, num_full_proj, angular_subsampling)]
    # projection file indices, we need to read in the projection in reverse order due to the portrait mode acquision
    projs_idx  = range(num_full_proj, 0, -angular_subsampling)

    n_pro = vecs.shape[0]

    # create the numpy array which will receive projection data from tiff files
    projs = np.zeros((num_full_proj, projs_rows, projs_cols), dtype=np.float32)

    # transformation to apply to each image, we need to get the image from
    # the way the scanner reads it out into to way described in the projection
    # geometry
    trafo = lambda image : np.transpose(np.flipud(image))

    # load flat-field and dark-fields
    # there are two flat-field images (taken before and after acquisition), we simply average them
    dark = trafo(imageio.imread(os.path.join(data_path_full, dark_name)))
    flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)

    for i, fn in enumerate(flat_name):
        flat[i] = trafo(imageio.imread(os.path.join(data_path_full, fn)))
    flat =  np.mean(flat,axis=0)

    # load projection data
    for i in range(num_full_proj):
        projs[i] = trafo(
                    imageio.imread(
                        os.path.join(data_path_full, projs_name.format(i))
                    )
                )
    projs = projs[::-1]

    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### pre-process data ###########################################################

    t = time.time();
    print('pre-process data', flush=True)
    # subtract the dark field, divide by the flat field, and take the negative log to linearize the data according to the Beer-Lambert law
    projs -= dark
    projs /= (flat - dark)
    np.log(projs, out=projs)
    np.negative(projs, out=projs)

    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### compute FDK reconstruction #################################################

    t = time.time();
    print('compute reconstruction', flush=True)

    # size of the reconstruction volume in voxels
    vol_sz  = 3*(50 * voxel_per_mm + 1,)
    # size of a cubic voxel in mm
    vox_sz  = 1/voxel_per_mm
    # numpy array holding the reconstruction volume
    vol_rec = np.zeros(vol_sz, dtype=np.float32)

    ref_rc = np.zeros(vol_sz, dtype=np.float32)
    ref_name = 'full_AGD_50_{:06}.tif'
    ref_name_2 = 'full_AGD_50_{:06}.tiff'
    for i in range(vol_sz[0]):
        try:
            ref_rc[i] = tiff.imread(exp_dir / 'Reconstructions' / ref_name.format(i))
        except FileNotFoundError:
            ref_rc[i] = tiff.imread(exp_dir / 'Reconstructions' / ref_name_2.format(i))

    # we need to specify the details of the reconstruction space to ASTRA
    # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
    # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
    vol_geom = astra.create_vol_geom(vol_sz)
    vol_geom['option']['WindowMinX'] = vol_geom['option']['WindowMinX'] * vox_sz
    vol_geom['option']['WindowMaxX'] = vol_geom['option']['WindowMaxX'] * vox_sz
    vol_geom['option']['WindowMinY'] = vol_geom['option']['WindowMinY'] * vox_sz
    vol_geom['option']['WindowMaxY'] = vol_geom['option']['WindowMaxY'] * vox_sz
    vol_geom['option']['WindowMinZ'] = vol_geom['option']['WindowMinZ'] * vox_sz
    vol_geom['option']['WindowMaxZ'] = vol_geom['option']['WindowMaxZ'] * vox_sz

    # we need to specify the details of the projection space to ASTRA
    # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
    proj_geom = astra.create_proj_geom('cone_vec', projs_rows, projs_cols, vecs)

    # register both volume and projection geometries and arrays to ASTRA
    vol_id  = astra.data3d.link('-vol', vol_geom, vol_rec)

    # # permute data to ASTRA convention
    # projs = np.transpose(projs, (1,0,2))
    # projs = np.ascontiguousarray(projs)

    proj_id = astra.data3d.link('-sino', proj_geom, np.ascontiguousarray(np.moveaxis(projs[sparse_indexes], 0, 1)))

    # finally, create an ASTRA configuration.
    # this configuration dictionary setups an algorithm, a projection and a volume
    # geometry and returns a ASTRA algorithm, which can be run on its own
    cfg_fdk = astra.astra_dict('FDK_CUDA')
    cfg_fdk['ProjectionDataId'] = proj_id
    cfg_fdk['ReconstructionDataId'] = vol_id
    cfg_fdk['option'] = {}
    cfg_fdk['option']['ShortScan'] = False
    alg_id = astra.algorithm.create(cfg_fdk)

    # run FDK algorithm
    astra.algorithm.run(alg_id, 1)

    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(np.round_(time.time() - t, 3), 'sec elapsed')

    ### save reconstruction ########################################################

    t = time.time();
    print('save results', flush=True)

    # low level plotting
    f, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(30,10))
    ax[0].imshow(vol_rec[vol_sz[0]//2,:,:], vmin=0)
    ax[1].imshow(vol_rec[:,vol_sz[1]//2,:])
    ax[2].imshow(vol_rec[:,:,vol_sz[2]//2])
    f.tight_layout()

    vol_rec.tofile(input_dir / sparse_reconstruction_dir / sparse_name_rc.format(walnut_id, orbit_id))
    projs.tofile(input_dir / sinogram_dir / sinogram_name.format(walnut_id, orbit_id))
    ref_rc.tofile(input_dir / dense_reconstruction_dir / ref_name_rc.format(walnut_id))

    shutil.rmtree(exp_dir)

    print(np.round_(time.time() - t, 3), 'sec elapsed')
