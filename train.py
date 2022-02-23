import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import pickle
import yaml
import tensorflow as tf
from tensorflow.keras import backend as K

from models import model_base
from keras_custom.callbacks import PredictionMonitor
from utils import load_config, data_loader, data_loader_gen, data_loader_gen_v2

# unclean but we have to eval heldout to decide whether rerun.
import evaluations

"""
Training logic for a given model.
    1. Load model
    2. Load data + Train model
    3. Save weights
"""

def execute(config_version, intermediate_input=True):
    config = load_config(config_version)
    model_name = config['model_name']
    results_path = f'results/{model_name}/{config_version}'
    if os.path.exists(results_path) is False:
        os.makedirs(results_path)

    # train and save model
    train_model(
        config=config, 
        intermediate_input=intermediate_input,
        results_path=results_path
    )    
                        
    
    # FIXME:
    # i.e. WARNING:tensorflow:7 out of the last 1030 calls to 
    # <function Model.make_predict_function.<locals>.predict_function at 0x7f30247d1a60> 
    # triggered tf.function retracing. 
    # Tracing is expensive and the excessive number of tracings could be due to 
    # (1) creating @tf.function repeatedly in a loop, 
    # (2) passing tensors with different shapes, 
    # (3) passing Python objects instead of tensors. 
    # For (1), please define your @tf.function outside of the loop. 
    # For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. 
    # For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.

    
    # multiple restarts when heldout result not ideal
    if config['heldout'] is not None:
        # first eval see if good.
        diff = eval_for_rerun(config)
        print(f'[Check] diff = {diff}')

        reruns = 0
        while diff > 0.1 and reruns < 5:
            
            # a new model is always init by `train_model`
            train_model(
                config=config, 
                intermediate_input=intermediate_input,
                results_path=results_path
            )  
            
            # eval the trained model again.
            diff = eval_for_rerun(config)
            reruns += 1
            print(f'[Check] reruns = {reruns}')
            

def train_model(config, intermediate_input, results_path):
    """
    Purpose:
    --------
        Train a model and save weights.
    
    inputs:
    -------
        train_data, val_data: 
              if finetine, both are list of array, i.e. train_data=[X_train, Y_val]
              if fulltrain, both are generators.
        train_steps, val_steps: Only used when data are generators.
    
    return:
    -------
        Saved model weights.
    """
    # model
    model, input_shape, preprocess_func = model_base(
        config_version=config['config_version'],
        intermediate_input=intermediate_input
    )
    model.summary()

    # data
    # when use intermediate input, load all at once.
    if intermediate_input is True:
        train_data, val_data = data_loader(
                                config=config,
                                input_shape=input_shape)
        train_steps = None
        val_steps = None
    # when use entire model, load by generator.
    else:
        print(f'\n[Warning] gen_v2 is used. Please rename.\n')
        train_data, train_steps, \
            val_data, val_steps = data_loader_gen_v2(
                                config=config,
                                preprocess_func=preprocess_func,
                                shuffle=True)

    # callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(
                    log_dir=f'{results_path}/log/')
    earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=config['patience'],
            mode='min', restore_best_weights=True)

    if config['heldout'] is None or config['stimulus_set'] in [6, '6']:
        callbacks = [tensorboard, earlystopping]
    else:
        # predictionmonitor = PredictionMonitor(config)
        # callbacks = [tensorboard, earlystopping, predictionmonitor]
        callbacks = [tensorboard, earlystopping]

    # print('[Check] reinitializing weights..')
    # model = reinitialize(x=train_data[0], model=model, scale=0.001)

    # fit model
    if intermediate_input is True:
        x = train_data[0]
        y = train_data[1]
        model.fit(
                x=x, 
                y=y,
                batch_size=config['batch_size'], 
                epochs=config['epochs'], 
                callbacks=callbacks,
                validation_data=val_data, 
                max_queue_size=10, 
                workers=10,
                use_multiprocessing=False,
                steps_per_epoch=train_steps,
                validation_steps=val_steps)
    else:
        model.fit(
                x=train_data, 
                batch_size=config['batch_size'], 
                epochs=config['epochs'], 
                callbacks=callbacks,
                validation_data=val_data, 
                max_queue_size=10, 
                workers=10,
                use_multiprocessing=False,
                steps_per_epoch=train_steps,
                validation_steps=val_steps)
    
    # save trained weights, del model.
    save_model(model, config, results_path)


def save_model(model, config, results_path):
    """
    if train == 'finetune': 
        save pred layer weights 

    if train == 'finetune-with-lowAttn': 
        save pred layer weights and attn weights
    """
    # make sure directory exists for the trained models.
    save_path = f'{results_path}/trained_weights'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path) 

    if config['train'] == 'finetune':
        # save pred layer weights as .pkl
        pred_weights = model.get_layer('pred').get_weights()
        with open(os.path.join(save_path, 'pred_weights.pkl'), 'wb') as f:
            pickle.dump(pred_weights, f)
        print('[Check] pred_weights saved.')

    elif config['train'] == 'finetune-with-lowAttn':
        # save pred layer weights as .pkl
        pred_weights = model.get_layer('pred').get_weights()
        with open(os.path.join(save_path, 'pred_weights.pkl'), 'wb') as f:
            pickle.dump(pred_weights, f)
        print('[Check] pred_weights saved.')

        # save attn layers weights as .npy (to be consistent with JointModel)
        attn_positions = config['attn_positions'].split(',')
        attn_weights = {}
        for attn_position in attn_positions:
            layer_attn_weights = \
                model.get_layer(
                    f'attn_factory_{attn_position}').get_weights()[0]
            attn_weights[attn_position] = layer_attn_weights

        np.save(f'{save_path}/attn_weights.npy', attn_weights)
        print(f'[Check] attn weights saved.')

    
    else:
        NotImplementedError()
    del model
    K.clear_session()
    

# FIXME: 
# RuntimeError: Detected a call to `Model.fit` inside a `tf.function`. 
# `Model.fit is a high-level endpoint that manages its own `tf.function`. 
# Please move the call to `Model.fit` outside of all enclosing `tf.function`s. 
# Note that you can call a `Model` directly on `Tensor`s inside a `tf.function` like: `model(x)`.

# @tf.function
# def loop_eval_for_rerun(
#         diff, reruns, 
#         config, intermediate_input, results_path,
#         diff_thr=0.1, reruns_thr=5,
#     ):
#     while diff > diff_thr and reruns < reruns_thr:
            
#         train_model(
#             config=config, 
#             intermediate_input=intermediate_input,
#             results_path=results_path
#         )  
        
#         # eval the trained model again.
#         diff = eval_for_rerun(config)
#         reruns += 1
#         print(f'[Check] reruns = {reruns}')


def eval_for_rerun(config):
    """
    Purpose:
    --------
        Execute evaluation to check heldout performance.
        If bad, rerun a few times due to unstable learning.
    """
    pred_reprs = evaluations.execute(
        config_version=config['config_version']
    )

    if config['stimulus_set'] not in [6, '6']:
        heldouts = ['000', '001', '010', '011',
                    '100', '101', '110', '111']
    else:
        heldouts = ['0000', '0001', '0010', '0011',
                    '0100', '0101', '0110', '0111',
                    '1000', '1001', '1010', '1011',
                    '1100', '1101', '1110', '1111']
    
    for i in range(len(heldouts)):
        heldout = heldouts[i]
        if config['heldout'] == heldout:
            trg_row = i
            break
    
    # convert '000' to [0, 0, 0]
    heldout_true = np.array([float(s) for s in config['heldout']])
    # take the correct row from (8, 3) or (16, 3) reprs
    heldout_pred = np.array([float(s) for s in pred_reprs[trg_row, :]])
    print(f'[Check] trg_row = {trg_row}')
    print(f'[Check] heldout_true = {heldout_true}')
    print(f'[Check] heldout_pred = {heldout_pred}')
    # if diff big, rerun this config
    diff = np.sum(np.abs(heldout_true - heldout_pred))
    return diff


def reinitialize(x, model, scale=0.2):
    """
    Purpose:
    --------
        reinitialize output layer weights such that
        output units follow a gaussian with mean=0.5

    Impl:
    -----
        Given a layer, we have training reprs x 
        and model intercepted at that layer + output layer. 
        We create gaussians with the same number of the output
        units and reverse engineer the required output layer
        weights that can produce these gaussians given x.

        In other words, 
            originally we have dot(x, w) = y
        Here, to get y_gaussian, 
            we do w = dot(inv(x), y_gaussian)

    inputs:
    -------
        x: training data (reprs from a specified layer)
        model: pre-trained model intercepted at specified layer
        scale: standard deviation of the gaussian.
    """
    # (n, 3)
    y_pred = model.predict(x, verbose=1)

    # (d, 3)
    weights = model.get_layer('pred').get_weights()[0]
    biases = model.get_layer('pred').get_weights()[1]   # bias are 0.

    # (n, 3), each dim follows N(0.5, sigma^2)
    np.random.seed(999)
    y_ideal_dim0 = np.clip(np.random.normal(loc=0.5, scale=scale, size=y_pred.shape[0]), 0, 1)
    y_ideal_dim1 = np.clip(np.random.normal(loc=0.5, scale=scale, size=y_pred.shape[0]), 0, 1)
    y_ideal_dim2 = np.clip(np.random.normal(loc=0.5, scale=scale, size=y_pred.shape[0]), 0, 1)
    y_ideal = np.array([y_ideal_dim0, y_ideal_dim1, y_ideal_dim2]).T

    # (d, n) dot (n, 3) = (d, 3)
    weights = np.dot(np.linalg.pinv(x), y_ideal)
    # NOTE: since biases are zero at init, y_pred = x * w = x * w + b
    # NOTE: we want x*w + b' = y_ideal, so b' = y_ideal - y_pred
    # biases = y_ideal - y_pred

    # swap init weights
    model.get_layer('pred').set_weights([weights, biases])
    return model


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()
    execute(args.config)