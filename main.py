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
import utils

"""
One script does it all.
    1. Prepare dataset (create either augmented reprs or images)
        -- data.py
    2. Train the model
        -- train.py
    3. Evaluate the trained model.
        -- evaluations.py
"""


def main_across_targets(stimulus_set, model_name, layer, run, args, full_test, heldout_test):
    """
    Wrapper for multiGPU training, one layer per GPU.
    For each layer, need to iterate through all heldouts.
    """
    if stimulus_set not in [6, '6']:
        heldouts = ['000', '001', '010', '011',
                    '100', '101', '110', '111', None]
    else:
        heldouts = ['0000', '0001', '0010', '0011',
                    '0100', '0101', '0110', '0111',
                    '1000', '1001', '1010', '1011',
                    '1100', '1101', '1110', '1111', None]
    for heldout in heldouts:
        config_version = f't{stimulus_set}.{model_name}.{layer}.{heldout}.run{run}'
        print(f'[Check] main running {config_version}')
        main(
            config_version=config_version, 
            mode=args.mode,
            full_test=full_test,
            heldout_test=heldout_test
        )


def main(config_version, mode, full_test, heldout_test):
    """
    inputs:
    -------
        full_test & heldout_test: collectors of eval results 
        default to None and only needed when visualise and compare
        which layer to use.
    """
    config_version = f'config_{config_version}'
    config = load_config(config_version)

    preprocessed_dir = config['preprocessed_dir']
    model_name = config['model_name']
    layer = config['layer']
    stimulus_set = config['stimulus_set']
    
    if mode == 'train':
        dataset_dir = f'stimuli/{preprocessed_dir}/{model_name}/' \
                  f'{layer}_reprs/task{stimulus_set}'
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
            heldout_test)
    K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode')
    parser.add_argument('--task', dest='task')
    args = parser.parse_args()

    start_time = time.time()
    # -------------------------------------------------------
    stimulus_sets = [args.task]
    runs = [1]
    # layers = ['fc2', 'flatten', 'block5_conv3', 'block5_conv2', 'block5_conv1', 'block4_pool', 'block3_pool']
    # model_name = 'vgg16'
    layers = ['layer_3']
    model_name = 'vit_b16'
    cuda_id_list = [0, 1, 2, 3, 4, 5, 6, 7]
    # -------------------------------------------------------

    for stimulus_set in stimulus_sets:

        for run in runs:
            # for eval easy comparison.
            full_test = defaultdict(list)
            heldout_test = defaultdict(list)

            # Enable multiGPU training (one layer per GPU)
            if args.mode == "train":
                args_list = []
                for layer in layers:
                    single_entry = {}
                    single_entry['stimulus_set'] = stimulus_set
                    single_entry['model_name'] = model_name
                    single_entry['layer'] = layer
                    single_entry['run'] = run
                    single_entry['args'] = args
                    single_entry['full_test'] = full_test
                    single_entry['heldout_test'] = heldout_test
                    args_list.append(single_entry)
                
                print(f"args_list:\n{args_list}")
                print(f"len(args_list): {len(args_list)}")
                utils.cuda_manager(
                    main_across_targets, args_list, cuda_id_list
                )
            
            # If eval, just use one GPU.
            elif args.mode == "eval":
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                for layer in layers:
                    main_across_targets(
                        stimulus_set, 
                        model_name, 
                        layer,
                        run, 
                        args, 
                        full_test, 
                        heldout_test
                    )
            else:
                raise ValueError(f'Unknown mode: {args.mode}')

            if args.mode == 'eval':
                for layer in layers:
                    print(
                        f'layer=[{layer}], ' \
                        f'full=[{np.mean(full_test[layer]):.4f}], ' \
                        f'heldout=[{np.mean(heldout_test[layer]):.4f}]'
                    )
    
    print(f'Total time: {time.time() - start_time:.2f}s')
