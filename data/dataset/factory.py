from typing import Optional, Sequence, Any, Iterable, Callable

import torch.utils.data as data

from .postprocessing import WalnutDataset, CorkDataset

def create_dataset(input_dir : str='',
                   input_file : str ='dataset.csv',
                   num_proj : int = 60,
                   patch_size : int = 256,
                   final_activation : str = 'Identity',
                   transforms: Optional[Callable] = None,
                   dataset_name: str='cork',
                   training : bool = True,
                   test : bool = False,
                   outputs: Iterable[str] = ['sparse_rc', 'reference_rc'],
                   mode : str='postprocessing',
                   **kwargs)-> data.Dataset:

    if mode == 'postprocessing':

        if dataset_name == 'cork':
            dataset = CorkDataset(input_dir,
                                  input_file=input_file,
                                  patch_size=patch_size,
                                  final_activation=final_activation,
                                  transforms=transforms,
                                  outputs=outputs,
                                  training=training,
                                  test=test,
                                  num_proj=num_proj,
                                  **kwargs)

        elif dataset_name == 'walnut':
            dataset = WalnutDataset(input_dir,
                                    input_file=input_file,
                                    patch_size=patch_size,
                                    final_activation=final_activation,
                                    transforms=transforms,
                                    outputs=outputs,
                                    training=training,
                                    test=test,
                                    num_proj=num_proj,
                                    **kwargs)

    return dataset