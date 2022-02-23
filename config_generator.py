import os
import yaml

"""
Automatically generate a bunch of config files 
by iterating through a range of params.
"""
# run = 1
# lr = 0.000003   # 3e-6
# run = 2
# lr = 0.00003      # 3e-5
# run = 3
# lr = 0.0003       # 3e-4
run = 1
lr = 0.003        # 3e-3
model_name = 'vgg16'
stimulus_sets = [1]
reg_strength = 0.
attn_positions = 'block4_pool'
train = 'finetune-with-lowAttn'
layers = [f'{attn_positions}']

dict_task1to5 = {'config_version': None,
                'model_name': model_name,
                'split_ratio': 0.2,
                'actv_func': 'sigmoid',
                'kernel_constraint': None, 
                'kernel_regularizer': None,
                'hyperbolic_strength': None,
                'lr': lr,
                'batch_size': 16,
                'epochs': 1000,
                'patience': 20,
                'run': run,
                'task': 'binary',
                'binary_loss': 'BCE',
                'train': train,
                'stimulus_set': None,
                'heldout': None,
                'layer': 'flatten',
                'preprocessed_dir': 'v3',
                'XY_dir': 'v3-binary',
                'size_per_class': 1023,
                'augment_seed': 42,
                'augmentations': {'rotate_range': 45,
                                  'shear_range': 15,
                                  'horizontal_flip': True,
                                  'vertical_flip': True},
                'low_attn_constraint': 'nonneg', 
                'attn_initializer': 'ones',
                'attn_regularizer': 'l1',
                'reg_strength': reg_strength,
                'noise_distribution': None,
                'noise_level': None,
                'attn_positions': attn_positions
                }

dict_task6 = {'config_version': None,
                'model_name': 'vgg16',
                'split_ratio': 0.2,
                'actv_func': 'sigmoid',
                'kernel_constraint': None, 
                'kernel_regularizer': None,
                'hyperbolic_strength': None,
                'lr': lr,
                'batch_size': 16,
                'epochs': 1000,
                'patience': 20,
                'run': run,
                'task': 'binary',
                'binary_loss': 'BCE',
                'train': 'finetune',
                'stimulus_set': None,
                'heldout': None,
                'layer': 'flatten',
                'preprocessed_dir': 'v3',
                'XY_dir': 'v3-binary',
                'min_shrink_rate': 0.25,
                'max_shrink_rate': 0.32,
                'num_zooms': 10,
                'num_locations': 10,
                'rotation_range': 20,
                'meta_seed': 42,          
                'noise_mode': 'gaussian'
                }

heldouts_task1to5 = [None,
                    '000', '001', '010', '011',
                    '100', '101', '110', '111']

heldouts_task6 = [None, 
                '0000', '0001', '0010', '0011',
                '0100', '0101', '0110', '0111',
                '1000', '1001', '1010', '1011',
                '1100', '1101', '1110', '1111']

for stimulus_set in stimulus_sets:

    # if task1-5, use the same default
    if stimulus_set not in [6, '6']:
        default_dict = dict_task1to5
        heldouts = heldouts_task1to5
    # if task6, use its own default  
    else:
        default_dict = dict_task6
        heldouts = heldouts_task6

    default_dict['stimulus_set'] = stimulus_set

    for layer in layers:
        default_dict['layer'] = layer

        for heldout in heldouts:
            default_dict['heldout'] = heldout
            run = default_dict['run']
            model_name = default_dict['model_name']

            if train == 'finetune-with-lowAttn':
                config_version = f'config_t{stimulus_set}.{model_name}.{layer}.{heldout}.run{run}-with-lowAttn'
            else:
                config_version = f'config_t{stimulus_set}.{model_name}.{layer}.{heldout}.run{run}'

            default_dict['config_version'] = config_version

            filepath = os.path.join(f'configs', f'{config_version}.yaml')
            with open(filepath, 'w') as yaml_file:
                yaml.dump(default_dict, yaml_file, default_flow_style=False)


