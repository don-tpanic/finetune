import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import load_config

"""
Some functions evaluating the finetuned lowAttn weights.
"""

def between_zero_percent_and_hyperparams():
    """
    Plot the relationship between zeroed out 
    attention weights after finetuning with L1 strength
    and learning rates (two hyper-params during finetune-lowAttn)
    """
    runs = range(1, 45)
    all_lr_finetune = []
    all_reg_strength = []
    all_zero_percent = []
    for run in runs:
        config_version = f'config_t1.vgg16.block4_pool.None.run{run}-with-lowAttn'
        config = load_config(config_version)
        lr_finetune = config['lr_finetune']
        reg_strength = config['reg_strength']
        dcnn_base = config['dcnn_base']
        all_lr_finetune.append(lr_finetune)
        all_reg_strength.append(reg_strength)

        path = f'results/{dcnn_base}/{config_version}/trained_weights'
        attn_position = config['low_attn_positions'].split(',')[0]
        attn_weights = np.load(f'{path}/attn_weights.npy', allow_pickle=True)
        layer_attn_weights = attn_weights.ravel()[0][attn_position]
        zero_percent = 1 - (
            len(np.nonzero(layer_attn_weights)[0]) / len(layer_attn_weights)
        )
        all_zero_percent.append(zero_percent)
    
    # plot a 2D scatter of x=l1 strength, y=lr rate, value=zero_percent
    fig, ax = plt.subplots(2, 1)
    # sort for plotting
    # l1
    order = np.argsort(all_reg_strength)
    all_reg_strength = np.array(all_reg_strength)[order]
    ax[0].scatter(
        range(len(all_zero_percent)),
        np.array(all_zero_percent)[order]
    )
    ax[0].set_xlabel('L1 strength (log scale)')
    ax[0].set_ylabel('percentage zero')
    ax[0].set_xticks(np.arange(len(all_reg_strength)))
    all_reg_strength[np.where(all_reg_strength == 0)[0]] = 1e-16  # avoid div by 0.
    ax[0].set_xticklabels(
        np.round(
            np.log(all_reg_strength), 1
        ), rotation='90'
    )
    ax[0].grid(True)

    # lr
    order = np.argsort(all_lr_finetune)
    all_lr_finetune = np.array(all_lr_finetune)[order]
    ax[1].scatter(
        range(len(all_zero_percent)), 
        np.array(all_zero_percent)[order]
    )
    ax[1].set_xlabel('learning rate (log scale)')
    ax[1].set_ylabel('percentage zero')
    ax[1].set_xticks(np.arange(len(all_lr_finetune)))
    ax[1].set_xticklabels(all_lr_finetune)
    ax[1].set_xticklabels(
        np.round(
            np.log(all_lr_finetune), 1
        ), rotation='90'
    )
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('zero_percent_l1_n_lr.png')


def finetuned_lowAttn(config_version):
    config = load_config(config_version=config_version)
    dcnn_base = config['dcnn_base']
    path = f'results/{dcnn_base}/{config_version}/trained_weights'
    attn_positions = config['low_attn_positions'].split(',')
    attn_weights = np.load(f'{path}/attn_weights.npy', allow_pickle=True)
    attn_weights = attn_weights.ravel()[0]

    for attn_position in attn_positions:
        layer_attn_weights = attn_weights[attn_position]
        nonzero_percentage = len(np.nonzero(layer_attn_weights)[0]) / len(layer_attn_weights)
        
        
        print(layer_attn_weights)
        print(f'nonzero_percentage = {nonzero_percentage}')
        print(f'mean = {np.mean(layer_attn_weights)}')
        print(f'std = {np.std(layer_attn_weights)}')
        print(f'max = {np.max(layer_attn_weights)}, min = {np.min(layer_attn_weights)}')
        print(f'sum = {np.sum(layer_attn_weights)}')

        plt.hist(layer_attn_weights)
        plt.savefig('attn_hist.png')


def pred_stability():
    tests = ['000', '001', '010', '011',
            '100', '101', '110', '111', None]
    for test in tests:

        dcnn_config_version1 = f't1.vgg16.block4_pool.{test}.run5seed42-with-lowAttn'
        dcnn_config_version2 = f't1.vgg16.block4_pool.{test}.run5seed42repeat1-with-lowAttn'

        dcnn_save_path1 = f'results/vgg16/config_{dcnn_config_version1}/trained_weights'
        with open(os.path.join(dcnn_save_path1, 'pred_weights.pkl'), 'rb') as f:
            pred_weights1 = pickle.load(f)

        dcnn_save_path2 = f'results/vgg16/config_{dcnn_config_version2}/trained_weights'
        with open(os.path.join(dcnn_save_path2, 'pred_weights.pkl'), 'rb') as f:
            pred_weights2 = pickle.load(f)


        import scipy.stats as st
        print(st.spearmanr(pred_weights1[0].flatten(), pred_weights2[0].flatten()))
    


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= f"-1"
    between_zero_percent_and_hyperparams()
    # pred_stability()

