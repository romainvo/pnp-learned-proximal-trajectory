from typing import Optional, Any, Mapping, Iterable, Union, List
from torch import Tensor

import torch
import torch.nn as nn

from .layers import create_upscaling_layer, TimestepEmbedding, Timesteps
from . import layers

class EncoderBlock(nn.Module):
    """
    # Parameters
        - in_channels (int): number of channels in input feature map
        - out_channels (int): number of channels in output feature map

    # Keyword arguments:
        - downscaling (bool)=True : False for center block
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 downscaling=True,
                 activation=nn.ReLU,
                 residual=False,
                 timestep_embed_dim: int=0):
        super(EncoderBlock, self).__init__()

        kernel_size = 3 if downscaling else kernel_size
        padding = 1 if downscaling else kernel_size // 2

        self.downscaling = downscaling
        if downscaling:
            self.downscaling_layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, 
                        kernel_size=kernel_size, 
                        stride=2, 
                        groups=in_channels,
                        padding=padding, 
                        bias=False),
                nn.BatchNorm2d(out_channels),
                activation(inplace=True)
            )

        self.residual = residual
        if self.residual:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(out_channels if downscaling else in_channels, out_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))

        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(out_channels if self.downscaling else in_channels, out_channels, 
                      kernel_size=kernel_size, 
                      stride=1, 
                      padding=padding, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )
        
        self.timestep_embed_dim = timestep_embed_dim
        if self.timestep_embed_dim > 0:
            self.timestep_connection = nn.Linear(in_features=self.timestep_embed_dim,
                                                 out_features=out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=padding, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True)
        
    def forward(self, x: Tensor, timestep_emb: Optional[Tensor]=None):

        if self.downscaling:
            x = self.downscaling_layer(x)

        shortcut = x
        
        x = self.conv_bn1(x)
        
        if timestep_emb is not None and self.timestep_embed_dim > 0:
            temb = nn.SiLU()(timestep_emb)
            temb = self.timestep_connection(temb)[..., None, None]
            x = x + temb
        
        x = self.conv2(x)
        if self.residual:
            x += self.skip_connection(shortcut)
        x = self.bn2(x)
        x = self.act(x)
        
        return x

class UNetEncoder(nn.Module):
    """
    # Parameters:
        - encoder_channels (list): list of int ordered from highest resolution to lowest
    
    # Keyword arguments:
        - activation (nn.Module): activation function
        - residual (bool)=False : use skip connections or not
    """
    def __init__(self, in_channels: int,
                       encoder_channels: Iterable[int]=[64,128,256,512,1024],
                       stem_size: int=3,
                       activation: nn.Module=nn.ReLU,
                       residual: bool=False,
                       block: nn.Module=EncoderBlock,
                       forward_features=True,
                       timestep_embed_dim: int=0):
        super(UNetEncoder, self).__init__()
        
        self.encoder_channels = encoder_channels 
        self.forward_features = forward_features

        self.stem_block = block(in_channels, encoder_channels[0], 
                                kernel_size=stem_size,
                                activation=activation, 
                                residual=residual,
                                downscaling=False,
                                timestep_embed_dim=timestep_embed_dim) 
        
        blocks = [block(in_channels, out_channels,
                        activation=activation, 
                        residual=residual,
                        timestep_embed_dim=timestep_embed_dim) 
                    for in_channels, out_channels
                    in zip(encoder_channels[:-1], encoder_channels[1:])
        ]
        self.encoder_blocks = nn.ModuleList(blocks)

        self.init_weights()
        
    def forward(self, x: Tensor, timestep_emb: Optional[Tensor]=None) -> Union[Tensor, List[Tensor]]:

        # ordered from highest resolution to lowest
        x = self.stem_block(x, timestep_emb=timestep_emb)
        if self.forward_features:
            features_list = [x]
        
        for block in self.encoder_blocks:
            x = block(x, timestep_emb=timestep_emb)
            if self.forward_features:
                features_list.append(x)

        if self.forward_features:                   
            return features_list
        else:
            return x

    def init_weights(self, nonlinearity : str = 'relu'):
        for name, m in self.named_modules():
            
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                if 'shuffle' not in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, torch.nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  

class DecoderBlock(nn.Module):
    """
    # Parameters
        - in_channels (int): number of channels in input feature map
        - skip_channels (int): number of channels in skip feature map
        - out_channels (int): number of channels in output feature map

    # Keyword arguments:
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
        - upscaling_layer=upconv (str): one of ['upconv', 'pixelshuffle', 'interpolation']
        - interpolation='bilinear' (str): interpolation mode in upscaling_layer function
    """
    def __init__(self, in_channels, skip_channels, out_channels,
                 scale_skip_connection=True, #encoder-decoder skip connection
                 residual=False,
                 activation=nn.ReLU,
                 upscaling_layer='upconv', 
                 interpolation='bilinear',
                 timestep_embed_dim: int=0):
        super(DecoderBlock, self).__init__()

        self.upscaling_layer = create_upscaling_layer(in_channels, 
                                                      scale_factor=2,
                                                      layer_name=upscaling_layer,
                                                      interpolation=interpolation,
                                                      activation=activation
        )

        self.scale_skip_connection = scale_skip_connection
        in_channels = in_channels + skip_channels if self.scale_skip_connection else in_channels

        self.residual = residual
        if self.residual:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=False))

        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )
        
        self.timestep_embed_dim = timestep_embed_dim
        if self.timestep_embed_dim > 0:
            self.timestep_connection = nn.Linear(in_features=self.timestep_embed_dim,
                                                 out_features=out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = activation(inplace=True)
        
    def forward(self, x: Tensor, skip: Tensor, timestep_emb: Optional[Tensor]) -> Tensor:
        x = self.upscaling_layer(x)
        if self.scale_skip_connection:
            x = torch.cat([x, skip], dim=1)

        shortcut = x

        x = self.conv_bn1(x)
        
        if timestep_emb is not None and self.timestep_embed_dim > 0:
            temb = nn.SiLU()(timestep_emb)
            temb = self.timestep_connection(temb)[..., None, None]
            x = x + temb
                                
        x = self.conv2(x)
        if self.residual:
            x += self.skip_connection(shortcut)
        x = self.bn2(x)
        x = self.act(x)
        
        return x
    
class UNetDecoder(nn.Module):
    """
    # Parameters:
        - encoder_channels (list): list of int ordered from highest resolution to lowest
        - decoder_channels (list): number of channels in decoder path
    
    # Keyword arguments:
        - activation (nn.Module)=nn.ReLU: activation function
        - residual (bool)=False : use skip connections or not
        - upscaling_layer=upconv (str): one of ['upconv', 'pixelshuffle', 'interpolation']
        - interpolation='bilinear' (str): interpolation mode in upscaling_layer function
    """
    def __init__(self, encoder_channels, decoder_channels, scale_skip_connections,
                 upscaling_layer='upconv', 
                 interpolation='bilinear',
                 activation=nn.ReLU,
                 residual=False,
                 block=DecoderBlock,
                 timestep_embed_dim: int=0):
        super(UNetDecoder, self).__init__()
        
        # Reverse list to start loop from lowest resolution block
        # from highest number of channels to lowest
        encoder_channels = encoder_channels[::-1] 

        in_channels_list = [encoder_channels[0]] + list(decoder_channels[:-1])
        
        blocks = [block(in_channels, skip_channels, out_channels,
                        scale_skip_connection=scale_connection,
                        activation=activation,
                        residual=residual,
                        upscaling_layer=upscaling_layer, 
                        interpolation=interpolation,
                        timestep_embed_dim=timestep_embed_dim) 
                    for in_channels, skip_channels, out_channels, scale_connection
                    in zip(in_channels_list, encoder_channels[1:], decoder_channels, scale_skip_connections)
        ]
        self.decoder_blocks = nn.ModuleList(blocks)

        self.init_weights()
        
    # Features ordered from highest resolution to lowest
    def forward(self, features: Iterable[Tensor], 
                      timestep_emb: Optional[Tensor]=None,
                      forward_features: bool=False) -> Union[Tensor, List[Tensor]]:

        results = []

        x = features[-1]
        # Reverse list of features to loop from lowest resolution to highest, 
        # and also remove features[-1] (because x = features[-1])
        features = features[-2::-1] 
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, features[i], timestep_emb)

            if forward_features:
                results.append(x)
        
        if forward_features:
            return results
                            
        return x

    def init_weights(self, nonlinearity : str = 'relu'):
        for name, m in self.named_modules():
            
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                if 'shuffle' not in name:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=nonlinearity)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, torch.nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)  
    
class UNet(nn.Module):
    """
    # Parameters:
        - backbone (str): backbone from timm
    
    # Keyword arguments:
    """
    def __init__(self,
                 in_channels=3, 
                 encoder_channels=[64,128,256,512,1024],
                 decoder_channels=[512,256,128,64,64],
                 scale_skip_connections=[1, 1, 1, 1, 1],
                 dropout=0.,
                 final_activation='Sigmoid', 
                 head_layer='RegressionHead',
                 **kwargs):
        super(UNet, self).__init__()


        residual = kwargs.pop('residual', False)
        activation = getattr(nn, kwargs.pop('activation', 'ReLU'))
            
        self.noise_level = bool(kwargs.pop('noise_level', 0.0))
        if self.noise_level > 0:
            print("****** NOISE LEVEL ******")
            
        self.timestep_dim = kwargs.pop('timestep_dim', 0)
        self.timestep_embed_dim = 0
        if self.timestep_dim > 0:
            self.timestep_projection = Timesteps(num_channels=self.timestep_dim,
                                                 max_period=1000)
            
            self.timestep_embed_dim = 4 * self.timestep_dim
            self.timestep_embedding = TimestepEmbedding(in_channels=self.timestep_dim,
                                                        time_embed_dim=self.timestep_embed_dim)
            
        in_channels = in_channels + 1 if self.noise_level else in_channels

        stem_size = kwargs.pop('stem_size', 3)
        self.encoder = UNetEncoder(in_channels, encoder_channels,
                                    stem_size=stem_size,
                                    activation=activation,
                                    residual=residual,
                                    block=EncoderBlock,
                                    timestep_embed_dim=self.timestep_embed_dim)
        self.encoder_channels = encoder_channels

        upscaling_layer = kwargs.pop('upscaling_layer', 'upconv')
        interpolation = kwargs.pop('interpolation', 'bilinear')

        scale_skip_connections = [bool(elt) for elt in scale_skip_connections]
        self.decoder = UNetDecoder(self.encoder_channels, 
                                   decoder_channels=decoder_channels,
                                   scale_skip_connections=scale_skip_connections,
                                   upscaling_layer=upscaling_layer, 
                                   interpolation=interpolation,
                                   residual=residual,
                                   activation=activation,
                                   block=DecoderBlock,
                                   timestep_embed_dim=self.timestep_embed_dim)
        
        self.sparse_autoencoding = kwargs.pop('sparse_autoencoding', False)
        if self.sparse_autoencoding:
            sparse_autoencoding_scale_skip_connections = kwargs.pop('sparse_autoencoding_scale_skip_connections', [0,0,0,0,0])
            self.sparse_decoder = UNetDecoder(self.encoder_channels,
                                              decoder_channels=decoder_channels,
                                              scale_skip_connections=sparse_autoencoding_scale_skip_connections,
                                              upscaling_layer=upscaling_layer, 
                                              interpolation=interpolation,
                                              residual=residual,
                                              activation=activation,
                                              block=DecoderBlock)


        last_channels = decoder_channels[-1]

        head = getattr(layers, head_layer)
        bias_free = kwargs.pop('bias_free', False)
        self.head = head(last_channels,
                         num_classes=1, # if not sigma_output else 2, 
                         dropout=dropout,
                         inter_channels=last_channels,
                         final_activation=final_activation,
                         bias_free=bias_free) 

        self.final_act = getattr(nn, final_activation)()

    def forward(self, x : Tensor, timesteps: Optional[Tensor]=None,
                                  residual_learning: bool=False, 
                                  noise_level: Optional[Tensor]=None) -> Tensor:

        timestep_emb = None
        if timesteps is not None and self.timestep_dim > 0:
            timestep_emb = self.timestep_projection(timesteps)
            timestep_emb = self.timestep_embedding(timestep_emb)
        elif timesteps is None and self.timestep_dim > 0:
            raise ValueError(f'self.timestep_dim={self.timestep_dim} and you forgot to pass argument `timesteps`')

        if self.noise_level:
            if noise_level is None:
                raise ValueError(f'self.noise_level={self.noise_level} and you forgot to pass argument `noise_level`')
            else:
                noise_level = torch.einsum('b,bcij->bcij', noise_level, torch.ones_like(x, device=x.device, dtype=x.dtype))
                input = torch.cat([x, noise_level], dim=1)
        else:
            input = x

        # Features ordered from highest resolution to lowest
        features = self.encoder(input, timestep_emb)
        features = self.decoder(features, timestep_emb)
                    
        results = self.final_act(self.head(features))
        if residual_learning:
            results = results + x
        
        return results

if __name__ == '__main__':

    from utils import count_parameters

    model = UNet(in_channels=1, 
                 encoder_channels=[32,32,64,64,128],
                 decoder_channels=[64,64,32,32],
                 dropout=0.,
                 final_activation='Sigmoid')
    setattr(model, 'head', None)

    print("Model #parameters : {}".format(count_parameters(model)))