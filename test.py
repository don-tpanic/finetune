import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import load_config


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

        dcnn_config_version1 = f't1.vgg16.block4_pool.{test}.run3'
        dcnn_config_version2 = f't1.vgg16.block4_pool.{test}.run4'

        dcnn_save_path1 = f'results/vgg16/config_{dcnn_config_version1}/trained_weights'
        with open(os.path.join(dcnn_save_path1, 'pred_weights.pkl'), 'rb') as f:
            pred_weights1 = pickle.load(f)

        dcnn_save_path2 = f'results/vgg16/config_{dcnn_config_version2}/trained_weights'
        with open(os.path.join(dcnn_save_path2, 'pred_weights.pkl'), 'rb') as f:
            pred_weights2 = pickle.load(f)


        import scipy.stats as st
        print(st.spearmanr(pred_weights1[0].flatten(), pred_weights2[0].flatten()))
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= f"0"
    config_version = 'config_t1.vgg16.block4_pool.None.run12-with-lowAttn'
    finetuned_lowAttn(config_version)