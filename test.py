import os 
import numpy as np
import pickle


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