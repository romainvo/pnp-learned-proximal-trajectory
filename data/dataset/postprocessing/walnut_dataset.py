from typing import Optional, Any

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from tqdm import tqdm

import math
import random

class WalnutDataset(data.Dataset):
    def __init__(self, input_dir, 
                       input_file='dataset.csv', 
                       patch_size=256,
                       final_activation='Sigmoid',
                       transforms=None, 
                       outputs=['sparse_rc', 'reference_rc'],
                       training=True,
                       test=False,
                       **kwargs):
        super(WalnutDataset, self).__init__()

        self.input_dir = Path(input_dir)
        self.patch_size = patch_size

        self.df = pd.read_csv(self.input_dir / input_file)

        self.final_activation = final_activation
        self.transforms = transforms
        self.training = training
        self.test = test

        self.outputs = outputs

        if 'split_set' in self.df:
            if training:
                self.df = self.df.loc[self.df.split_set == 'train']
            elif not training and not test:
                self.df = self.df.loc[self.df.split_set == 'validation']
            else:
                self.df = self.df.loc[self.df.split_set == 'test']

        self.num_voxels = self.df.iloc[0].num_voxels

        # Remove upper and bottom slice which contains weird non-dense material
        axial_center_crop = kwargs.pop('axial_center_crop', False)

        self.sample_list = []
        for row in self.df.itertuples():
            if axial_center_crop:
                for slice_index in range(100,501-100):
                    self.sample_list += [(row.id, slice_index)] 
            else:
                for slice_index in range(501):
                    self.sample_list += [(row.id, slice_index)]

        self.create_memmap()

        self.dataset_size = kwargs.pop('dataset_size', 3200)
        self.sample_indexes = self.random_sampler()

        self.center_crop = kwargs.pop('center_crop', False)

        print(f'center_crop={self.center_crop}')

        self.no_clip_val = kwargs.pop('no_clip_val', False)

        self.indi = kwargs.pop('indi', False)
        self.indi_eps = kwargs.pop('indi_eps', 0.0)
        self.num_timesteps = kwargs.pop('num_timesteps', 100)
        self.pnp = kwargs.pop('pnp', False)
        self.noise_level = kwargs.pop('noise_level', 0.0) > 0.

    def random_sampler(self):
        rng = np.random.default_rng()
        while True:
            shuffled_indexes = rng.permutation(list(range(len(self.sample_list))))
            for idx in shuffled_indexes:
                yield idx

    def create_memmap(self):
        self.sparse_rcs = dict()
        self.reference_rcs = dict()
        print("Initializing memory-mapping for each volume....", end='\n')
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            sample_id = row.id

            self.reference_rcs[sample_id] \
            = np.memmap(self.input_dir / row.reconstruction_file, 
                        dtype='float32', 
                        mode='r', 
                        shape=(row.number_of_slice, row.num_voxels, row.num_voxels))

            self.sparse_rcs[sample_id] \
            = np.memmap(self.input_dir / row.sparse_reconstruction_file,
                        dtype='float32', 
                        mode='r', 
                        shape=(row.number_of_slice, row.num_voxels, row.num_voxels))

    def __getitem__(self, index):
        index = next(self.sample_indexes)
        row_id, slice_index = self.sample_list[index]
        row = self.df.loc[self.df.id == row_id]
        row = row.iloc[0]

        if self.center_crop:
            h_offset = np.random.randint(50, row.num_voxels - self.patch_size - 50)
            w_offset = np.random.randint(50, row.num_voxels - self.patch_size - 50)
        else:
            h_offset = np.random.randint(row.num_voxels - self.patch_size)
            w_offset = np.random.randint(row.num_voxels - self.patch_size)

        reference_slice = self.reference_rcs[row_id][slice_index, 
                                                     h_offset:h_offset+self.patch_size,
                                                     w_offset:w_offset+self.patch_size]
        reference_slice = np.copy(reference_slice)

        if self.pnp:
            L_max = 0.502464 # FIXME: hard-coded
            sigma = (L_max * 10 / 255) * np.random.random()
            noise = sigma * np.random.randn(*reference_slice.shape).astype(np.float32)    
            sparse_slice = reference_slice + noise
            sigma = torch.tensor(sigma, dtype=torch.float32)
        else:
            sparse_slice = self.sparse_rcs[row_id][slice_index, 
                                                h_offset:h_offset+self.patch_size,
                                                w_offset:w_offset+self.patch_size]
            sparse_slice = np.copy(sparse_slice)
        
        if not self.no_clip_val:
            reference_slice = np.clip(reference_slice, 0., None)
            sparse_slice = np.clip(sparse_slice, 0., None)

        if self.indi:
            s = random.random()
            t = math.sin(s * math.pi / 2)
            sparse_slice[:] = (1 - t) * reference_slice + t * sparse_slice
            # add noise
            sparse_slice += math.sqrt(t) * self.indi_eps * np.random.normal(0, 1.0, sparse_slice.shape)
            
            t = torch.tensor(self.num_timesteps * t, dtype=torch.float32)

        assert (reference_slice.dtype is np.dtype('float32')), \
            print(reference_slice.dtype, reference_slice.max(), row)
        assert (sparse_slice.dtype is np.dtype('float32')), \
            print(sparse_slice.dtype, sparse_slice.max(), row)

        ########################

        if self.final_activation == 'Tanh':
            sparse_slice = (2 * sparse_slice - 1.).astype(np.float32)
            reference_slice = (2 * reference_slice - 1.).astype(np.float32)

        inputs = dict(sparse_rc=sparse_slice, reference_rc=reference_slice)

        results = self.transforms(**inputs)

        for key in self.outputs:
            if key in self.df:
                results[key] = row[key]

        if self.indi:
            return [results[key] for key in self.outputs] + [t]
        elif self.noise_level:
            return [results[key] for key in self.outputs] + [sigma]
        return [results[key] for key in self.outputs]    

    def __len__(self):

        return self.dataset_size 
    
class WalnutInferenceDataset(data.Dataset):
    def __init__(self, input_dir, 
                       input_file='dataset.csv', 
                       patch_stride=64,
                       patch_size=256,
                       final_activation='Sigmoid',
                       transforms=None, 
                       outputs=['sparse_rc', 'reference_rc'],
                       split_set: str='test',
                       **kwargs):
        super(WalnutInferenceDataset, self).__init__()

        self.input_dir = Path(input_dir)
        self.patch_stride = patch_stride
        self.patch_size = patch_size

        self.df = pd.read_csv(self.input_dir / input_file)

        self.final_activation = final_activation
        self.transforms = transforms

        self.outputs = outputs

        if 'split_set' in self.df:
            self.df = self.df.loc[self.df.split_set == split_set]

        self.num_voxels = self.df.iloc[0].num_voxels
        self.number_of_slice = self.df.iloc[0].number_of_slice

        detector_offsets = [offset for offset in np.arange(0, self.num_voxels, patch_stride) if offset + patch_size <= self.num_voxels]
        if len(detector_offsets) > 0:
            if detector_offsets[-1] + patch_size != self.num_voxels:
                detector_offsets.append(self.num_voxels - patch_size)

            self.offsets = list(itertools.product(detector_offsets, detector_offsets))

            self.slice_counts = np.zeros((self.num_voxels, self.num_voxels), dtype=np.float32)
            for h_offset in detector_offsets:
                for w_offset in detector_offsets:
                    self.slice_counts[h_offset:h_offset + patch_size, w_offset:w_offset + patch_size] += 1
        else:
            self.offsets = [(0, 0)]
            self.slice_counts = np.ones((self.num_voxels, self.num_voxels), dtype=np.float32)
            self.padding_left = math.ceil((self.patch_size - self.num_voxels) / 2)
            self.padding_right = self.patch_size - self.num_voxels - self.padding_left
            
        self.create_memmap()

        print(f"Initializing sample list with 1st row of self.df : {self.df.iloc[0].id}")
        self.sample_list = None
        self.current_row_idx = None
        self.current_row = self.update_sample_list(0)
        print(f"Num samples = {len(self)}")

        self.normalization = kwargs.pop('normalization', None)
        self.no_clip_val = kwargs.pop('no_clip_val', False)

    def create_memmap(self):
        self.sparse_rcs = dict()
        self.reference_rcs = dict()
        print("Initializing memory-mapping for each volume....", end='\n')
        for row in tqdm(self.df.itertuples(), total=len(self.df)):
            sample_id = row.id

            self.reference_rcs[sample_id] \
            = np.memmap(self.input_dir / row.reconstruction_file, 
                        dtype='float32', 
                        mode='c', 
                        shape=(row.number_of_slice, row.num_voxels, row.num_voxels))

            self.sparse_rcs[sample_id] \
            = np.memmap(self.input_dir / row.sparse_reconstruction_file,
                        dtype='float32', 
                        mode='c', 
                        shape=(row.number_of_slice, row.num_voxels, row.num_voxels))

    def update_sample_list(self, row_idx, row_id: Optional[Any]=None):
        assert (row_idx < len(self.df)), print(f"self.df only has {len(self.df)} rows and row_idx = {row_idx}")

        self.sample_list = []

        if row_id is not None:
            row = self.df.loc[self.df.id == row_id]
            row_idx = row.index[0]
            self.current_row_idx = row_idx
            row = row.iloc[0]
        else:
            self.current_row_idx = row_idx
            row = self.df.iloc[row_idx]

        print(f"Updating sample list with {row_idx}-th row of self.df : {row.id}")

        for slice_index in range(row.offset_top, row.offset_bottom):
            for (h_offset, w_offset) in self.offsets:
                self.sample_list += [(row.id, slice_index, h_offset, w_offset)]

        return row

    def __getitem__(self, index):

        row_id, slice_index, h_offset, w_offset = self.sample_list[index]
        row = self.current_row

        if self.patch_size > self.num_voxels:
            padding_left = math.ceil((self.patch_size - self.num_voxels) / 2)
            padding_right = self.patch_size - self.num_voxels - padding_left
            
            reference_slice = np.pad(self.reference_rcs[row_id][slice_index],
                                     ((padding_left, padding_right), (padding_left, padding_right)),
                                     mode='constant')
            sparse_slice = np.pad(self.sparse_rcs[row_id][slice_index],
                                  ((padding_left, padding_right), (padding_left, padding_right)),
                                  mode='constant')
        else:
            reference_slice = self.reference_rcs[row_id][slice_index, 
                                                        h_offset:h_offset+self.patch_size,
                                                        w_offset:w_offset+self.patch_size]
            sparse_slice = self.sparse_rcs[row_id][slice_index, 
                                                h_offset:h_offset+self.patch_size,
                                                w_offset:w_offset+self.patch_size]

        if not self.no_clip_val:
            reference_slice = np.clip(reference_slice, 0., None)
            sparse_slice = np.clip(sparse_slice, 0., None)

        assert (reference_slice.dtype is np.dtype('float32')), \
            print(reference_slice.dtype, reference_slice.max(), row)
        assert (sparse_slice.dtype is np.dtype('float32')), \
            print(sparse_slice.dtype, sparse_slice.max(), row)

        ########################

        if self.final_activation == 'Tanh':
            sparse_slice = (2 * sparse_slice - 1.).astype(np.float32)
            reference_slice = (2 * reference_slice - 1.).astype(np.float32)

        inputs = dict(sparse_rc=sparse_slice, reference_rc=reference_slice)

        results = self.transforms(**inputs)

        for key in self.outputs:
            if key in self.df:
                results[key] = row[key]

        return [results[key] for key in self.outputs] + [row_id, slice_index, h_offset, w_offset]   

    def __len__(self):

        return len(self.sample_list)