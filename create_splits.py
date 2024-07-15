import random
import pandas as pd
import numpy as np
from zipfile import ZipFile
import io
import copy

random.seed(42)

if __name__ == '__main__':

    sample_list = list(range(1,43))
    random.shuffle(sample_list)

    row_list = []

    base_row = {
            'id': 1,
            'num_full_proj': 1200,
            'num_voxels': 501,
            'sinogram_width': 768,
            'sinogram_height': 972,
            'offset_top': 0,
            'offset_bottom': 501,
            'number_of_slice':501,
            'sinogram_file': 'sinogram/Walnut{}_pos2.raw',
            'reconstruction_file': 'dense_reconstruction/Walnut{}_GT.raw',
            'sparse_reconstruction_file': 'sparse_reconstruction_50/Walnut{}_pos2_FDK.raw',
            'trajectory_file': 'trajectory/Walnut{}_pos2.npy',
            'split_set': 'train',
            'is_log': True,
            'sinogram_format': 'DMM'
    }

    for i in range(42):
        row = copy.deepcopy(base_row)
        id = sample_list.pop()

        row['id'] = id
        row['sinogram_file'] = row['sinogram_file'].format(id)
        row['reconstruction_file'] = row['reconstruction_file'].format(id)
        row['sparse_reconstruction_file'] = row['sparse_reconstruction_file'].format(id)
        row['trajectory_file'] = row['trajectory_file'].format(id)

        with ZipFile(f'Walnut{id}.zip', 'r') as file_in:
            trajectory_object = file_in.read(f'Walnut{id}/Projections/tubeV2/scan_geom_corrected.geom')
            
            trajectory = np.loadtxt(io.BytesIO(trajectory_object), skiprows=0)
            np.save(f'trajectory/Walnut{id}_pos2.npy', trajectory)

        if i < 30:
            row['split_set'] = 'train'
        elif i < 34:
            row['split_set'] = 'validation'
        else:
            row['split_set'] = 'test'

        row_list.append(row)

    df = pd.DataFrame.from_dict(row_list)
    df.to_csv('dataset_50p.csv', index=False)

