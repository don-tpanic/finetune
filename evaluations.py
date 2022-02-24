import os
import pickle
import argparse
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

from utils import load_config, data_loader, produce_orig_reprs
from models import model_base

"""
Post-finetune evaluations

Compare correlation between DCNN representations of the stimuli
to the experimenter codings of the stimuli. This is to measure if 
we have achieved sensible representations of the stimuli from DCNN
using whatever approach which would enable us to move onto the next step.
"""

def execute(config_version, full_test=None, heldout_test=None):
    config = load_config(config_version)
    reprs = original_stimuli_final_coordinates(config)
    
    # only for visual comparison of layers.
    if full_test is not None:
        write_reprs_to_table(reprs, config, full_test, heldout_test)
        
    return reprs


def original_stimuli_final_coordinates(config):
    """
    Output trained model's prediction of given stimulus set.
    
    inputs:
    -------
        config
    """
    # Load model achitecture.
    config_version = config['config_version']
    model_name = config['model_name']
    stimulus_set = config['stimulus_set']

    model, _, preprocess_func = model_base(config_version=config_version)
    model.summary()

    # load the trained prediction layer weights.
    save_path = f'results/{model_name}/{config_version}/trained_weights'
    if config['train'] == 'finetune':
        with open(os.path.join(save_path, 'pred_weights.pkl'), 'rb') as f:
            pred_weights = pickle.load(f)
        model.get_layer('pred').set_weights(pred_weights)
        print(f'[Check] pred_weights loaded.')
    
    # load both trained pred layer weights and attn weights.
    elif config['train'] == 'finetune-with-lowAttn':
        with open(os.path.join(save_path, 'pred_weights.pkl'), 'rb') as f:
            pred_weights = pickle.load(f)
        model.get_layer('pred').set_weights(pred_weights)
        print(f'[Check] pred_weights loaded.')

        attn_positions = config['attn_positions'].split(',')
        attn_weights = np.load(f'{save_path}/attn_weights.npy', allow_pickle=True)
        attn_weights = attn_weights.ravel()[0]
        reg_strength = config['reg_strength']

        for attn_position in attn_positions:
            layer_attn_weights = attn_weights[attn_position]
            print(f'attn weights (attn_position), l1={reg_strength} = \n{layer_attn_weights}')
            nonzero_percentage = len(np.nonzero(layer_attn_weights)[0]) / len(layer_attn_weights)
            print(f'nonzero_percentage = {nonzero_percentage}')
            model.get_layer(
                f'attn_factory_{attn_position}').set_weights([layer_attn_weights])
            print(f'[Check] have set attn weights after {attn_position}')

    # Load original images and grab reprs
    # (n, d) e.g. (8, 3) or (16, 4)

    # NOTE: hacky way to be compatible when finetuned with low-attn.
    if 'lowAttn' not in config['config_version']:
        reprs, _ = produce_orig_reprs(
            model=model, 
            preprocess_func=preprocess_func,
            stimulus_set=stimulus_set)
    else:
        reprs, _ = produce_orig_reprs(
            model=model, 
            preprocess_func=preprocess_func,
            stimulus_set=stimulus_set,
            config=config)
    
    # release RAM
    del model
    K.clear_session()
    
    reprs = np.matrix.round(reprs, 2)
    heldout = config['heldout']
    print(f'\nheldout = {heldout}')
    print(reprs)
    print('----------------------\n')
    return reprs


def write_reprs_to_table(reprs, config, full_test, heldout_test):
    """
    Automating script that writes predicted reprs (binary codings)
    across layers into one table for easy comparison.
    """
    heldout = config['heldout']
    layer = config['layer']
    stimulus_set = config['stimulus_set']
    if stimulus_set not in ['6', 6]:
        y_trues = ['000', '001', '010', '011',
                    '100', '101', '110', '111']
    else:
        # Forward compatible for stimulus set 6.
        pass
    
    # if not heldout, fail 1 stimulus is fail.
    if heldout is None:
        score = 1
        for i in range(len(y_trues)):
            
            y_true = np.array(
                [float(s) for s in y_trues[i]]
            )
            y_pred = reprs[i]
            diff = np.sum(np.abs(y_true - y_pred))
            
            # a single fail will flip the flag.
            if diff > 0.1:
                score = 0
                break
        full_test[layer].append(score)
            
    # if heldout, just check the heldout
    else:
        score = 1
        for i in range(len(y_trues)):
            
            y_true = y_trues[i]
            
            if config['heldout'] == y_true:
                trg_row = i
                break
        
        y_true = np.array([float(s) for s in y_trues[i]])   
        y_pred = np.array([float(s) for s in reprs[trg_row, :]])
        diff = np.sum(np.abs(y_true - y_pred))
        if diff > 0.1:
            score = 0
        heldout_test[layer].append(score)
            
    print(f'[Results] heldout={heldout}, layer={layer}, score={score}')
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    execute(args.config)
