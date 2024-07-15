import torch.utils.data as data

def create_dataloader(dataset : data.Dataset, batch_size : int,
                      num_workers : int=0,
                      trainval : bool=False,
                      shuffle : bool=True,
                      drop_last : bool=True,
                      pin_memory : bool=False,
                      prefetch_factor : int=2,
                      sampler=None,
                      transforms=None,
                      persistent_workers: bool=True) -> data.DataLoader:

    if transforms is not None:
        dataset.transforms = transforms

    if sampler is not None:
        dataloader = data.DataLoader(dataset, batch_size,
                                     num_workers=num_workers,
                                     sampler=sampler,
                                     drop_last=drop_last,
                                     pin_memory=pin_memory,
                                     prefetch_factor=prefetch_factor)
    else:
        dataloader = data.DataLoader(dataset, batch_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle,
                                     drop_last=drop_last,
                                     pin_memory=pin_memory,
                                     persistent_workers=persistent_workers,
                                     prefetch_factor=prefetch_factor)

    return dataloader