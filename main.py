import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras import backend as K

import data
import train
import evaluations
from utils import load_config

"""
One script does it all.
    1. Prepare dataset (create either augmented reprs or images)
        -- data.py
    2. Train the model
        -- train.py
    3. Evaluate the trained model.
        -- evaluations.py
"""

def main(config_version, mode, full_test, heldout_test, zero_percent_dict):
    """
    inputs:
    -------
        full_test & heldout_test: collectors of eval results 
        default to None and only needed when visualise and compare
        which layer to use.
    """
    config_version = f'config_{config_version}'
    config = load_config(config_version)

    XY_dir = config['XY_dir']
    model_name = config['dcnn_base']
    layer = config['layer']
    stimulus_set = config['stimulus_set']

    if 'lowAttn' in config_version:
        first_attn_position = config['low_attn_positions'].split(',')[0]
        print(f'[Check] first_attn_position = {first_attn_position}')
        layer = first_attn_position
    
    if mode == 'train':
        dataset_dir = f'resources/{XY_dir}/{model_name}/' \
                  f'{layer}/task{stimulus_set}/X.npy'
        # if no dataset, create dataset 
        if not os.path.exists(dataset_dir):
            print('[Check] Preparing dataset..')
            data.execute(config)
        
        print('[Check] Start training..')
        train.execute(config_version)

    elif mode == 'eval':
        evaluations.execute(
            config_version, 
            full_test, 
            heldout_test, 
            zero_percent_dict
        )
    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode')
    parser.add_argument('-t', '--task', dest='task', default='1')
    parser.add_argument('-g', '--gpu', dest='gpu_index')
    parser.add_argument('-a', '--attn', dest='with_lowAttn', default=None)
    parser.add_argument('-b', '--run_begin', dest='run_begin', type=int)
    parser.add_argument('-e', '--run_end', dest='run_end', type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{args.gpu_index}"
    stimulus_sets = [args.task]
    with_lowAttn = args.with_lowAttn  # finetune with low-attn
    
    # -------------------------------------------------------
    runs = range(args.run_begin, args.run_end+1)
    # layers = ['fc2', 'fc1', 'flatten', 'block5_conv3', 'block5_conv2', 'block5_conv1', 'block4_pool', 'block3_pool']
    layers = ['block4_pool']
    model_name = 'vgg16'
    # -------------------------------------------------------

    for stimulus_set in stimulus_sets:

        # for eval easy comparison.
        full_test = defaultdict(lambda: defaultdict(list))
        heldout_test = defaultdict(lambda: defaultdict(list))
        zero_percent_dict = defaultdict(lambda: defaultdict(list))

        for run in runs:
            # get sum of runtime for all heldouts per layer
            layer_runtime_collector = []
            
            for layer in layers:
                # restart timer for a new layer
                start_time = time.time()

                if stimulus_set not in [6, '6']:
                    heldouts = ['000', '001', '010', '011',
                                '100', '101', '110', '111', None]
                else:
                    heldouts = ['0000', '0001', '0010', '0011',
                                '0100', '0101', '0110', '0111',
                                '1000', '1001', '1010', '1011',
                                '1100', '1101', '1110', '1111', None]
                for heldout in heldouts:
                    if with_lowAttn is None:
                        config_version = f't{stimulus_set}.{model_name}.{layer}.{heldout}.run{run}'
                    else:
                        config_version = f't{stimulus_set}.{model_name}.{layer}.{heldout}.run{run}-with-lowAttn'
                    print(f'[Check] training {config_version}')
                    main(
                        config_version=config_version, 
                        mode=args.mode,
                        full_test=full_test,
                        heldout_test=heldout_test,
                        zero_percent_dict=zero_percent_dict
                    )
                    
                if args.mode == 'train':
                    end_time = time.time()
                    # in hrs.
                    duration = (end_time - start_time) / 3600.
                    layer_runtime_collector.append(duration)
                    
                    np.save(
                        f'results/{model_name}/config_{config_version}/{layer}_runtime_collector.npy', 
                        layer_runtime_collector
                    )
        
        # Formatted output for examination.
        if args.mode == 'eval':
            for run in runs:
                for layer in layers:
            
                    # For same run, params same, so just use None.
                    if with_lowAttn is None:
                        config_version = f'config_t{stimulus_set}.{model_name}.{layer}.None.run{run}'
                    else:
                        config_version = f'config_t{stimulus_set}.{model_name}.{layer}.None.run{run}-with-lowAttn'

                    config = load_config(config_version)
                    lr = config['lr_finetune']
                    reg_strength = config['reg_strength']
                    print(
                        f'------------------------------------------'
                        f'\nrun={run}, lr={lr}, l1={reg_strength}\n '
                        f'layer=[{layer}], ' \
                        f'full=[{np.mean(full_test[run][layer]):.4f}], ' \
                        f'heldout=[{np.mean(heldout_test[run][layer]):.4f}], ' \
                        f'zero%={zero_percent_dict[run][layer]}'
                    )
