import argparse
import yaml

def nullable_string(val):
    if not val:
        return None
    return val

def get_config_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    return config_parser

def get_arg_parser():

    parser = argparse.ArgumentParser(description = "Deep Tomography training")

    parser.add_argument('--input_dir', default='', type=str,
                        help='Path to the directory that contains the --input_file. All the paths referenced in the input file are relative to this path.')
    parser.add_argument('--checkpoint_file', default='', type=nullable_string,
                        help='Path to the checkpoint file to resume training from.')
    parser.add_argument('--input_file', default='dataset.csv', type=str,
                        help='Path to the input file containing the paths to the data')
    parser.add_argument('--output_base', default='train', type=str)
    parser.add_argument('--output_dir', default='', type=str, 
                        help='path to output folder (default: none, will save directly at the root of output_base/)')
    parser.add_argument('--output_suffix', default='', type=nullable_string)
    parser.add_argument('--log_interval', default=60, type=int,
                        help='Interval for logging training status')
    parser.add_argument('--log_name', default='train', type=nullable_string,
                        help='Name of the log file')
    parser.add_argument('--dataset_name', default='cork', type=str)

    parser.add_argument('--validation_interval', default=1, type=int)
    parser.add_argument('--dataset_size', default=3200, type=int,
                        help='Number of batch steps per epoch')

    parser.add_argument('--noise_level', default=0., type=float)

    parser.add_argument('--residual_learning', action='store_true',
                        help='Use residual learning --> the network learns the residual between the input and the ground truth')
    parser.add_argument('--no-residual_learning', action='store_false', dest='residual_learning')

    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--no-center_crop', action='store_false', dest='center_crop')
    parser.add_argument('--axial_center_crop', action='store_true')
    parser.add_argument('--no-axial_center_crop', action='store_false', dest='axial_center_crop')
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--patch_stride', default=32, type=int)
    parser.add_argument('--final_activation', default='Sigmoid', type=str)
    parser.add_argument('--no-augmentation', action='store_false', dest='augmentation')
    parser.add_argument('--augmentation', action='store_true')

    parser.add_argument('--model_name', default='unet', type=str)
    
    parser.add_argument('--indi', action='store_true', help='Set up InDI data pre-processing (interp between sparse and dense)')
    parser.add_argument('--indi_eps', default=0.0, type=float)
    parser.add_argument('--pnp', action='store_true', help='Set up PnP data pre-processing (input = GT + noise)')
    parser.add_argument('--time_sampling_bias', default=3., type=float)
    parser.add_argument('--jacobian_spectral_norm_finetuning', action='store_true', help='Use spectral norm regularization during a finetuning phase')
    parser.add_argument('--no-jacobian_spectral_norm_finetuning', action='store_false', dest='jacobian_spectral_norm_finetuning')
    parser.add_argument('--jacobian_spectral_norm_power_steps', default=5, type=int)
    parser.add_argument('--jacobian_spectral_norm_weight', default=1e-5, type=float)
    parser.add_argument('--jacobian_spectral_norm_finetuning_epochs', default=10, type=int)

    parser.add_argument('--upscaling_layer', default='transposeconv', type=str)
    parser.add_argument('--activation', default='ReLU', type=str)
    parser.add_argument('--skip_connection', action='store_true')
    parser.add_argument('--stem_size', default=3, type=int)
    parser.add_argument('--encoder_channels', default=[32,32,64,64,128], type=int, nargs='+')
    parser.add_argument('--decoder_channels', default=[64,64,32,32], type=int, nargs='+')
    parser.add_argument('--scale_skip_connections', default=[1,1,1,1], type=int, nargs='+')
    parser.add_argument('--timestep_dim', default=0, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--bias_free', action='store_true')
    parser.add_argument('--no-bias_free', action='store_false', dest='bias_free')

    parser.add_argument('--loss', default='mae', type=str)
    parser.add_argument('--eval_metric', default='mse', type=str)
    parser.add_argument('--num_proj', default=60, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dampening', default=0., type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--optimizer_eps', default=1e-8, type=float)
    parser.add_argument('--amsgrad', action='store_true')
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    
    parser.add_argument('--num_timesteps', default=1000, type=int)

    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.999, type=float)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--init_lr', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--num_epochs_restart', default=-1, type=int)

    parser.add_argument('--num_warmup_epochs', default=5, type=int)
    parser.add_argument('--warmup_start', default=1e-8, type=float)
    parser.add_argument('--lr_scheduler', default='ReduceLROnPlateau', type=str)
    parser.add_argument('--milestones', default=[20,50,80], type=int, nargs='+')
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--plateau_threshold', default=5e-2, type=float)

    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--no-pin_memory', dest='pin_memory', action='store_false')
    parser.add_argument('--memmap', action='store_true')
    parser.add_argument('--no-memmap', dest='memmap', action='store_false')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
    parser.add_argument('--no-amp', action='store_false', dest='amp', help='Enable automatic mixed precision training')

    parser.add_argument('--clip_grad_norm', default=0., type=float) 
    parser.add_argument('--clip_grad_value', default=0., type=float) 

    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--cpu', action='store_true')

    return parser

def parse_args():

    config_parser = get_config_parser()
    parser = get_arg_parser()

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text