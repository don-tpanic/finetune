import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_custom.generators import gen, gen_w_heldout

"""
Shared utility functionalities.
    `load_config`: load configuration file

    `produce_orig_reprs`:
        1. Given the original stimuli (or the backgrounded stimuli for task6),
        2. Load a pretrained model, compute layer (user given) representations.

    `data_loader`:
        1. Take the layer reprs, stack them into X, Y (giant matrices)
        2. Save them and load back in later for training.
        3. The stacking/saving thing only needs to do once.
"""

def load_config(config_version):
    with open(os.path.join('configs', f'{config_version}.yaml')) as f:
        config = yaml.safe_load(f)
    print(f'[Check] Loading [{config_version}]')
    return config


def produce_orig_reprs(model, preprocess_func, stimulus_set, return_images=False):
    """
    Purpose:
    --------
        Produce given layer's representations of 8 or 16
        stimuli with no data augmentation only preprocessed.

    inputs:
    -------
        model: A specified model capped at some layer
        preprocess_func: Model-specific preprocessing routine
        stimulus_set: ..
        return_images: default False (return model predictions)
                        If True, return the preprocessed original
                        images.

    returns:
    --------
        reprs: layer activations for all images.
                reprs will have shape (N, D)
        dataset: A tf Dataset which produces original images later 
                    used for visual examination.
    """
    data_dir = f'stimuli/original/task{stimulus_set}'
    print(f'[Check]: using data from {data_dir}')

    batch_size = len(os.listdir(data_dir))
    print(f'[Check] batch_size={batch_size}')

    # this loads original images
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                shuffle=False,
                image_size=(224, 224),
                batch_size=batch_size)

    # this loads model-specific processed images.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=preprocess_func)
    generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=False)
    if return_images is False: 
        reprs = model.predict(generator)
    else:
        reprs, _ = next(generator)

        ### TEST gen match real image. ###
        # fig, ax = plt.subplots(1, reprs.shape[0])
        # for i in range(reprs.shape[0]):
        #     ax[i].imshow(reprs[i]/255.)
        #     ax[i].set_title(f'type[{i}]')
        # plt.savefig('testGenmatch.pdf')
        # exit()
        ### ###

    return reprs, dataset


def data_loader(config, input_shape, seed=42):
    """
    Purpose:
    --------
        - Load train/val datasets.
        - Also we have the option to load dataset for heldout training,
          in other words, a class will be held out during training.
        - This data_loader is compatible for task1-6.
    
    Impl:
    -----
        Load and stack all data-points as a giant matrix to be 
        shuffled and splitted later for training/validation. 
        The entire matrix will be saved so next time we do not 
        have to stack one data-point at a time but loading in
        the entire matrix at once for training/validation.

    inputs:
    -------
        config: ..
        input_shape: this is used to set the empty array to enable concat.
                     and should be the fc1 output size.
        seed: control randomness in train/val split
    """
    XY_dir = config['XY_dir']
    stimulus_set = config['stimulus_set']
    split_ratio = config['split_ratio']
    model_name = config['model_name']
    layer = config['layer']

    # we only stack the data once, once saved we can load off the disk.
    if os.path.exists(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}/X.npy'):
        print(f'[Check] Loading pre-saved X and Y from {XY_dir}/{model_name}/{layer}/task{stimulus_set}/')
        X = np.load(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}/X.npy')
        Y = np.load(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}/Y.npy')
        print(f'[Check] X, Y shape = {X.shape}, {Y.shape}')

    # first time, we stack and save dataset into disk.
    else:
        # if not the Current Biology set, we have 3 features.
        if stimulus_set not in [6, '6']:
            orig2binary = {
                        '000': [0, 0, 0], 
                        '001': [0, 0, 1],
                        '010': [0, 1, 0],
                        '011': [0, 1, 1],
                        '100': [1, 0, 0],
                        '101': [1, 0, 1],
                        '110': [1, 1, 0],
                        '111': [1, 1, 1]}
        # 4 features.
        else:
            orig2binary = {'0000': [0,0,0,0],
                            '0001': [0,0,0,1],
                            '0010': [0,0,1,0],
                            '0011': [0,0,1,1],
                            '0100': [0,1,0,0],
                            '0101': [0,1,0,1],
                            '0110': [0,1,1,0],
                            '0111': [0,1,1,1],
                            '1000': [1,0,0,0],
                            '1001': [1,0,0,1],
                            '1010': [1,0,1,0],
                            '1011': [1,0,1,1],
                            '1100': [1,1,0,0],
                            '1101': [1,1,0,1],
                            '1110': [1,1,1,0],
                            '1111': [1,1,1,1]}

        X = np.empty(input_shape)
        if config['stimulus_set'] not in ['6', 6]:
            Y = np.empty(3)
        else:
            Y = np.empty(4)
        mapping = orig2binary

        preprocessed_dir = config['preprocessed_dir']
        data_dir = f'stimuli/{preprocessed_dir}/{model_name}/{layer}_reprs/task{stimulus_set}/'
        print(f'[Check] Stacking reprs from {data_dir}')
        for stimulus_type in sorted(os.listdir(data_dir)):
            print(f'[Check] Stacking stimulus [{stimulus_type}]')
            y = mapping[stimulus_type]
            for fname in os.listdir(os.path.join(data_dir, stimulus_type)):
                fpath = os.path.join(data_dir, stimulus_type, fname)
                x = np.load(fpath)
                X = np.vstack((X, x))
                Y = np.vstack((Y, y))
                
        X = X[1:, :]
        Y = Y[1:, :]
        print(f'[Check] X.shape={X.shape}')
        print(f'[Check] Y.shape={Y.shape}')

        # save the stacked dataset
        if not os.path.exists(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}'):
            os.makedirs(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}')
        np.save(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}/X.npy', X)
        np.save(f'resources/{XY_dir}/{model_name}/{layer}/task{stimulus_set}/Y.npy', Y)
        print('[Check] saved X, Y.')


    # if heldout, we slice a subset of the X and Y
    # based on the stimulus type we want to hold out.
    heldout_class = config['heldout']
    if config['heldout'] is not None:
        if stimulus_set not in [6, '6']:
            num_sample_per_class = 1024
            if heldout_class == '000':
                a = num_sample_per_class * 0
            if heldout_class == '001':
                a = num_sample_per_class * 1
            if heldout_class == '010':
                a = num_sample_per_class * 2
            if heldout_class == '011':
                a = num_sample_per_class * 3
            if heldout_class == '100':
                a = num_sample_per_class * 4
            if heldout_class == '101':
                a = num_sample_per_class * 5
            if heldout_class == '110':
                a = num_sample_per_class * 6
            if heldout_class == '111':
                a = num_sample_per_class * 7
            
            heldout_indices = np.arange(a, a+num_sample_per_class)
            X = np.delete(X, heldout_indices, axis=0)
            Y = np.delete(Y, heldout_indices, axis=0)
            print(f'[Check] holding out [{heldout_class}]')
        else:
            # because task6 has different number of samples
            # the slices need to set up differently
            num_sample_per_class = 400
            if heldout_class == '0000':
                a = num_sample_per_class * 0
            if heldout_class == '0001':
                a = num_sample_per_class * 1
            if heldout_class == '0010':
                a = num_sample_per_class * 2
            if heldout_class == '0011':
                a = num_sample_per_class * 3
            if heldout_class == '0100':
                a = num_sample_per_class * 4
            if heldout_class == '0101':
                a = num_sample_per_class * 5
            if heldout_class == '0110':
                a = num_sample_per_class * 6
            if heldout_class == '0111':
                a = num_sample_per_class * 7

            if heldout_class == '1000':
                a = num_sample_per_class * 8
            if heldout_class == '1001':
                a = num_sample_per_class * 9
            if heldout_class == '1010':
                a = num_sample_per_class * 10
            if heldout_class == '1011':
                a = num_sample_per_class * 11
            if heldout_class == '1100':
                a = num_sample_per_class * 12
            if heldout_class == '1101':
                a = num_sample_per_class * 13
            if heldout_class == '1110':
                a = num_sample_per_class * 14
            if heldout_class == '1111':
                a = num_sample_per_class * 15

            heldout_indices = np.arange(a, a+num_sample_per_class)
            X = np.delete(X, heldout_indices, axis=0)
            Y = np.delete(Y, heldout_indices, axis=0)
            print(f'[Check] holding out [{heldout_class}]')

    X_train, X_val, \
        Y_train, Y_val = train_test_split(
                            X, Y, 
                            test_size=split_ratio, 
                            random_state=seed)
    print(f'[Check] Training data: {X_train.shape}')
    print(f'[Check] Validation data: {X_val.shape}')
    return (X_train, Y_train), (X_val, Y_val)


def data_loader_gen(config, preprocess_func, shuffle, seed=42):
    """Use generator as data loader for training"""

    preprocessed_dir = config['preprocessed_dir']
    model_name = config['model_name']
    stimulus_set = config['stimulus_set']
    directory = f'stimuli/{preprocessed_dir}/{model_name}/processed_imgs/task{stimulus_set}'

    # TODO. Not ideal but does the trick of loading .npy images
    # from `gen.py`
    if stimulus_set not in ['6', 6]:
        class_mode = 'binary_feat3'
        preprocess_func = None
    else:
        class_mode = 'binary_feat4'

    print(f'[Check] Generator loading data from {directory}')
    train_data = gen.DirectoryIterator(
            directory=directory,
            class_mode=class_mode,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            seed=seed,
            validation_split=config['split_ratio'],
            subset='training',
            preprocessing_function=preprocess_func)

    train_data = label_converter(train_data, stimulus_set)
    train_steps = train_data.compute_step_size()

    val_data = gen.DirectoryIterator(
            directory=directory,
            class_mode=class_mode,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            seed=seed,
            validation_split=config['split_ratio'],
            subset='validation',
            preprocessing_function=preprocess_func)

    val_data = label_converter(val_data, stimulus_set)
    val_steps = val_data.compute_step_size()

    print(f'[Check] train/val steps={train_steps},{val_steps}')
    return train_data, train_steps, val_data, val_steps


def data_loader_gen_v2(config, preprocess_func, shuffle, seed=42):
    """
    v2: supports heldout training

    Use generator as data loader for training
    """
    preprocessed_dir = config['preprocessed_dir']
    model_name = config['model_name']
    stimulus_set = config['stimulus_set']
    directory = f'stimuli/{preprocessed_dir}/{model_name}/processed_imgs/task{stimulus_set}'
    heldout_class = config['heldout']

    if stimulus_set not in ['6', 6]:
        class_mode = 'binary_feat3'
        preprocess_func = None
        all_classes = ['000', '001', '010', '011',
                       '100', '101', '110', '111']
    else:
        class_mode = 'binary_feat4'
        NotImplementedError()
    all_classes_indices = dict(zip(all_classes, range(len(all_classes))))

    # NOTE(ken), this is a hacky bit where we maually construct the dict
    # such that heldout can be done.
    if heldout_class is None:
        classes = all_classes
        class_indices = all_classes_indices
    else:
        classes = [c for c in all_classes if c!= heldout_class]
        class_indices = {}
        for c in classes:
            class_indices[c] = all_classes_indices[c]
    print(f'[Check] class_indices = {class_indices}')

    print(f'[Check] Generator loading data from {directory}')
    train_data = gen_w_heldout.DirectoryIterator(
            directory=directory,
            class_mode=class_mode,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            seed=seed,
            validation_split=config['split_ratio'],
            subset='training',
            preprocessing_function=preprocess_func,
            classes=classes,
            class_indices=class_indices)

    train_data = label_converter(train_data, stimulus_set)
    train_steps = train_data.compute_step_size()

    val_data = gen_w_heldout.DirectoryIterator(
            directory=directory,
            class_mode=class_mode,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            seed=seed,
            validation_split=config['split_ratio'],
            subset='validation',
            preprocessing_function=preprocess_func,
            classes=classes,
            class_indices=class_indices)

    val_data = label_converter(val_data, stimulus_set)
    val_steps = val_data.compute_step_size()

    print(f'[Check] train/val steps={train_steps},{val_steps}')
    return train_data, train_steps, val_data, val_steps


def label_converter(generator, stimulus_set):
    """
    Purpose:
    --------
        Only used when 
            train == 'fulltrain' & task == 'binary'
        Or train == 'funtune' & task == 'binary' & stimulus_set = 6
        This is because the default generator produces labels 
        as `sparse` ints whereas for binary prediction we want 
        to predict 0/1. 
    Impl:
    -----
        We have to intercept the default generators and manually 
        substitute the y labels using a mapping.
    """
    if stimulus_set not in ['6', 6]:
        class2binary = {0: [0, 0, 0], 
                        1: [0, 0, 1],
                        2: [0, 1, 0],
                        3: [0, 1, 1],
                        4: [1, 0, 0],
                        5: [1, 0, 1],
                        6: [1, 1, 0],
                        7: [1, 1, 1]}
    # This is when task=6, we have 4 features as targets.
    else:
        class2binary = {0: [0,0,0,0],
                        1: [0,0,0,1],
                        2: [0,0,1,0],
                        3: [0,0,1,1],
                        4: [0,1,0,0],
                        5: [0,1,0,1],
                        6: [0,1,1,0],
                        7: [0,1,1,1],
                        8: [1,0,0,0],
                        9: [1,0,0,1],
                        10: [1,0,1,0],
                        11: [1,0,1,1],
                        12: [1,1,0,0],
                        13: [1,1,0,1],
                        14: [1,1,1,0],
                        15: [1,1,1,1]}

    mapped_classes = []
    for i, label in enumerate(generator.classes):
        temp = class2binary[label]
        mapped_classes.append(temp)
    generator.classes = mapped_classes
    return generator



if __name__ == '__main__':
    pass


