import torch.nn as nn

""" 
Weight decay setting for biases and norm layers implemented by / Copyright 2020 Ross Wightman at https://github.com/rwightman/pytorch-image-models
"""
def set_bn_weight_decay(model, weight_decay= 0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': weight_decay},
            {'params': decay}]
            
def get_optim_policies(model, init_lr, weight_decay):
    normal_weight = []
    normal_bias = []
    lr5_weight = []
    lr10_bias = []
    bn = []
    custom_ops = []

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
            ps = list(m.parameters())

            if 'classifier' in name and 'final' in name:
                lr5_weight.append(ps[0])
                if len(ps) == 2:
                    lr10_bias.append(ps[1])
            else:
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, torch.nn.GroupNorm)):
            bn.extend(list(m.parameters()))

        elif len(m._modules) == 0:
            if len(list(m.parameters())) > 0:
                raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    return [
        {'params': normal_weight, 'lr': init_lr, 'weight_decay': weight_decay,
         'name': "normal_weight"},
        {'params': normal_bias, 'lr': 2 * init_lr, 'weight_decay': 0,
         'name': "normal_bias"},
        {'params': bn, 'lr': init_lr, 'weight_decay': 0,
         'name': "BN scale/shift"},
        {'params': custom_ops, 'lr': init_lr, 'weight_decay': weight_decay,
         'name': "custom_ops"},
        # for fc
        {'params': lr5_weight, 'lr': 2.5 * init_lr, 'weight_decay': weight_decay,
         'name': "lr5_weight"},
        {'params': lr10_bias, 'lr': 5 * init_lr, 'weight_decay': 0,
         'name': "lr10_bias"}
    ]