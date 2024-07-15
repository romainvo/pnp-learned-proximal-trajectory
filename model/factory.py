import torch
from typing import Tuple, Sequence

from .unet import UNet

def create_model(model_name : str,
                 skip_connection : bool = True,
                 activation : str = 'ReLU',
                 final_activation : str = 'Identity',
                 encoder_channels : Sequence[int] = [32,32,64,64,128],
                 decoder_channels : Sequence[int] = [64,64,32,32],
                 upscaling_layer : str = 'transposeconv',
                 interpolation : str = 'bilinear',
                 **kwargs):

    in_channels = 1

    if model_name == 'unet':
        model = UNet(in_channels=in_channels, 
                     encoder_channels=encoder_channels,
                     decoder_channels=decoder_channels,
                     upscaling_layer=upscaling_layer, 
                     interpolation=interpolation,
                     activation=activation,
                     residual=skip_connection,
                     final_activation=final_activation,
                     **kwargs)
        print("Encoder channels :", encoder_channels)
        print("Decoder channels :", decoder_channels)   

    return model