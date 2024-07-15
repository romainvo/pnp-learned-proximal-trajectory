# Plug-and-Play Learned Proximal Trajectory for 3D Sparse-View X-Ray Computed Tomography

Code for the paper "Plug-and-Play Learned Proximal Trajectory for 3D Sparse-View X-Ray Computed Tomography". 

## Pre-requisites

You will need the following libraries:
- `pytorch=1.11`
- `pytigre=2.1.0`
- `pandas`
- `matplotlib`
- `tifffile`
- `pyyaml`
- `tqdm`
- `timm`
- `PIL`
- `piqa`

The `torch` and `TIGRE` code is run with cuda 11.3.
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Instructions on how to install the `TIGRE` toolbox for 3D X-ray Cone-Beam Computed Tomography can be found here https://github.com/CERN/TIGRE. In this code we use the version `2.1.0`.

## Learned Proximal Trajectory

### Saving a pre-defined trajectory

Running this command will save a pre-defined optimization trajectory on the Walnut-CBCT dataset with $\eta = 1/L$, $\lambda=5$ and $K=200$ iterations. You need to specify: 
- `--input_dir=<input_dir>`, the location where `dataset_50p.csv` is stored.
- `--output_dir=<output_dir>`, where you want the logs to be saved.
- `--trajectory_output_dir=<trajectory_output_dir>`, where the pre-defined optimization trajectory is going to be saved.

```
python -m tigre_rc  \
    --dataset_name='walnut' \
    --input_dir='<input_dir>' \
    --input_file='dataset_50p.csv' \
    --split_set='train' \
    --acquisition_id='8' \
    --output_base='train' \
    --output_dir='<output_dir>' \
    --init_mode='FDK' \
    --ideal_denoiser \
    --num_full_proj=1200 \
    --num_proj=50 \
    --block_size=50 \
    --init_lr=1.0 \
    --lambda_reg=5.0 \
    --fista \
    --reconstruction_alg='proximal_splitting' \
    --n_iter=200 \
    --save_trajectory \
    --trajectory_output_dir=<trajectory_output_dir> \
```

### Training

Running this command will train a *Learned Proximal Operator* on the previously saved optimization trajectory. This is the configuration used for the results in Tab.1 .
You need to specify:
- `--input_dir=<trajectory_output_dir>`, i.e. where you saved the optimization trajectory earlier.
- `--output_dir=<output_dir>`, where you want the logs to be saved.

Options:
- `--amp` will train the model in mixed precision.

This will save a `summary.csv` logs file as well as images during training and a checkpoint `last.pth.tar` to use during inference.

```
python -m train_trajectory  \
    --batch_size=32 \
    --patch_size=256 \
    --model_name='unet' \
    --no-residual_learning \
    --skip_connection \
    --encoder_channels 32 32 64 64 128 \
    --decoder_channels 64 64 32 32 \
    --scale_skip_connections 1 1 1 1 \
    --upscaling_layer='transposeconv_nogroup' \
    --input_dir=<trajectory_output_dir> \
    --output_base='train' \
    --output_dir=<output_dir> \
    --activation='SiLU' \
    --loss='mse' \
    --dropout=0.5 \
    --bias_free \
    --clip_grad_norm=1e-2 \
    --ema \
    --ema_decay=0.999 \
    --timestep_dim=128 \
    --stem_size=5 \
    --num_proj=50 \
    --num_epochs=1000 \
    --num_warmup_epochs=5 \
    --lr_scheduler='CosineAnnealingLR' \
    --init_lr=1e-4 \
    --min_lr=1e-8 \
    --weight_decay=1e-6 \
    --dataset_size=6400 \
    --log_interval=100 \
    --dataset_name='walnut' \
    --validation_interval=50 \
    --final_activation='Identity' \
    --augmentation \
    --optimizer='adam' \
    --drop_last \
    --num_workers=8 \
    --pin_memory \
    --memmap \
    --amp \
```

### Inference

Running this command will launch a PnP-PGD algorithm with our Learned Proximal Trajectory procedure. We use the parameters $\eta=1/L$, $\lambda=10$ and $K=500$.
To run this command, you will need two available GPUs, for speed and also because communication between `torch` and `TIGRE` does not work very well on a single GPU. You will need to specifiy:
- `--input_dir=<input_dir>`, where `dataset_50p.csv` is located
- `--output_dir=<output_dir>`, where you want the logs and the reconstruction files to be stored.
- `--reg_checkpoint=<path/to/last.pth.tar>`, path to the checkpoint file of the model generated after training.
- `--benchmark` means that it will run the procedure on every sample in the `split_set` selected.
- `--reg_batch_size` you can adapt this parameter depending on the VRAM available on your GPU.

```
python -m tigre_rc \
    --benchmark \
    --dataset_name='walnut' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --split_set='validation' \
    --output_base='train' \
    --output_dir=<output_dir> \
    --num_proj=50 \
    --reconstruction_alg='proximal_splitting' \
    --init_lr=1. \
    --init_mode='FDK' \
    --pnp \
    --fista \
    --reg_checkpoint=<path/to/last.pth.tar> \
    --ema \
    --stopping='num_iter' \
    --timesteps \
    --ckpt_fidelity \
    --ckpt_reconstruction \
    --ckpt_criterion \
    --lambda_reg=10.0 \
    --block_size=50 \
    --n_iter=501 \
    --reg_batch_size=16 \
    --output_suffix='something' \
```

### Evaluation

You evaluate PSNR and SSIM on a PnP experiment by running the following command. You will need to specifiy:
- `--input_dir=<input_dir>`, where `dataset_50p.csv` is located
- `--bench_location=<output_dir>`, where you saved the reconstruction files during the inference procedure.

This will generate a `results_<split_set>.csv` file with the metrics for each reconstruction and the average over the set.

```
python -m test  \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --dataset_name='walnut' \
    --file_format='{acquisition_id}_criterion.raw' \
    --split_set='validation' \
    --memmap \
    --bench_location=<output_dir> \
```

## PnP-$\alpha$PGD

The procedure to train a Gaussian denoising network for PnP-PGD and run the inference is very similar to our scheme.

### Training

```
python -m train_postp  \
    --batch_size=32 \
    --patch_size=256 \
    --model_name='unet' \
    --residual_learning \
    --skip_connection \
    --encoder_channels 32 32 64 64 128 \
    --decoder_channels 64 64 32 32 \
    --scale_skip_connections 1 1 1 1 \
    --upscaling_layer='transposeconv_nogroup' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --output_base='train' \
    --output_dir=<output_dir> \
    --activation='SiLU' \
    --dropout=0.5 \
    --bias_free \
    --clip_grad_norm=1e-2 \
    --pnp \
    --ema \
    --axial_center_crop \
    --center_crop \
    --stem_size=5 \
    --num_proj=50 \
    --init_lr=1e-4 \
    --min_lr=1e-8 \
    --weight_decay=1e-6 \
    --dataset_size=6400 \
    --num_epochs=1000 \
    --num_warmup_epochs=5 \
    --log_interval=100 \
    --dataset_name='walnut' \
    --validation_interval=10 \
    --final_activation='Identity' \
    --augmentation \
    --loss='mse' \
    --optimizer='adam' \
    --lr_scheduler='CosineAnnealingLR' \
    --drop_last \
    --num_workers=8 \
    --pin_memory \
    --memmap \
```

### Inference

```
python -m tigre_rc \
    --benchmark \
    --dataset_name='walnut' \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --split_set='validation' \
    --output_base='train' \
    --output_dir=<output_dir> \
    --num_proj=50 \
    --reconstruction_alg='proximal_splitting' \
    --init_lr=1. \
    --init_mode='FDK' \
    --pnp \
    --fista \
    --reg_checkpoint=<path/to/last.pth.tar> \
    --ema \
    --stopping='num_iter' \
    --ckpt_fidelity \
    --ckpt_reconstruction \
    --ckpt_criterion \
    --lambda_reg=10.0 \
    --block_size=50 \
    --n_iter=501 \
    --reg_batch_size=16 \
    --output_suffix='something' \
```

### Test

```
python -m test  \
    --input_dir=<input_dir> \
    --input_file='dataset_50p.csv' \
    --dataset_name='walnut' \
    --file_format='{acquisition_id}_criterion.raw' \
    --split_set='validation' \
    --memmap \
    --bench_location=<output_dir> \
```