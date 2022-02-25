import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from models import model_base
from utils import load_config, produce_orig_reprs
# from create_task6_images import load_n_resize_img, augment_img


"""
Prepare dataset. Since there are only 8 original stimuli for each task set,
in order to train the model to learn predicting the binary features, we need
to create augmented samples out of the 8 original.

What this script does is to 
    1. create augmented (seeded) samples, 
    2. load them into a pretrained model to grab
        given layer output representations 
    3. save those representations later used as 
        input for training.

`save_processed_data`: 
    Used for task1-5

`save_preprocessed_data_54`
    Used for task6
"""

def execute(config):

    # ---------------------------
    save_as_reprs = True
    save_as_imgs = False
    return_images = save_as_imgs
    # ---------------------------

    print(f'\n[Check] save_as_reprs = {save_as_reprs}')
    print(f'[Check] save_as_imgs = {save_as_imgs}\n')

    model, _, preprocess_func = model_base(
        dcnn_base=config['dcnn_base'], 
        layer=config['layer'], 
        train='none',
    )  

    if config['stimulus_set'] not in [6, '6']:
        save_processed_data(model=model, 
                            config=config,
                            dcnn_base=config['dcnn_base'], 
                            preprocess_func=preprocess_func, 
                            size_per_class=config['size_per_class'],
                            augment_seed=config['augment_seed'],
                            augmentations=config['augmentations'],
                            config_version=config['config_version'],
                            save_as_reprs=save_as_reprs,
                            layer=config['layer'],
                            save_as_imgs=save_as_imgs,
                            return_images=return_images)
    else:
        save_processed_data_54(
                            model=model, 
                            config=config, 
                            preprocess_func=preprocess_func)
    
    del model
    K.clear_session()


def save_processed_data(model, config,
                        dcnn_base, 
                        preprocess_func, 
                        size_per_class, 
                        augment_seed,
                        augmentations,
                        config_version,
                        save_as_reprs=False,
                        layer='flatten',
                        save_as_imgs=True,
                        return_images=True):
    """
    Purpose:
    --------
        Prepare dataset for model fitting.

    Impl:
    -----
        We follow the total number of items specified by `size_per_class`. 
        To make sure we get different augmentations, we use a fixed set of seeds. 
        There are size_per_class total seeds.
        For each seed, we load in the original 8 images
            1) we first do data augmentations and saved either reprs or images
                into 8 folders.
            2) we then save the un-augmented version reprs or images into the 8 folders.
                Since we no longer (config>=10) upsample the original 8, we use this trick to make
                sure we only save the original 8 once by not keeping track of their seeds.
        
    inputs:
    -------
        model: Model that will produce activations from one layer before fc2.
        config: ..
        dcnn_base: ..
        preprocess_func: ..
        size_per_class: total number of data-points per class
        augment_seed: random seed for data augmentation
        augmentations: yaml parameters specify data augmentations
    """
    stimulus_set = config['stimulus_set']
    classes = ['000', 
               '001', 
               '010', 
               '011', 
               '100', 
               '101', 
               '110', 
               '111']

    # The layer output of the original 8 stimuli (preprocessed)
    # We want to include them into total samples.
    x_orig, _ = produce_orig_reprs(
                            model=model, 
                            preprocess_func=preprocess_func,
                            stimulus_set=config['stimulus_set'],
                            return_images=return_images)

    # With a `size_per_class` in mind, we sample some seeds 
    # Where each seed controls one set of augmentations for the original stimuli.
    np.random.seed(augment_seed)
    seeds = np.random.choice(np.arange(1,2000), 
                             size=int(size_per_class),
                             replace=False)

    # Each seed controls one way of augmentation.
    # Each seed produces 8 augmented images, 1 for each class.
    # Each augmented images/reprs then in order saved in folders.
    for seed in seeds:
        print(f'[Check] generator grabs seed[{seed}]')
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=augmentations['rotate_range'],
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=None,
            shear_range=augmentations['shear_range'],
            zoom_range=0.0,
            channel_shift_range=0.0,
            fill_mode="nearest",
            cval=0.0,
            horizontal_flip=augmentations['horizontal_flip'],
            vertical_flip=augmentations['vertical_flip'],
            rescale=None,
            preprocessing_function=preprocess_func,
            data_format=None,
            validation_split=0.0,
            dtype=None)
        # load in the original stimuli and ready for augmentations.
        generator = datagen.flow_from_directory(
                f'stimuli/original/task{stimulus_set}',
                target_size=(224, 224),
                batch_size=8,
                class_mode='sparse',
                shuffle=False,
                seed=seed)
        
        # Make sure labels match when loading in.
        # print('[Check]', next(generator)[1], generator.class_indices)
        # (8, dims) 
        preprocessed_dir = config['preprocessed_dir']

        # Two options
        # 1. save the layer outputs as training set
        # 2. save the augmented images as training set
        # The first option saves a lot of computation.
        if save_as_reprs:
            # x = model.predict(generator)
            
            images, y = next(generator)  # NOTE(ken) confirmed match
            x = model(images)
            
            ftype = f'{layer}_reprs'

        elif save_as_imgs:
            x, y = next(generator)
            print(x[0].shape)
            ftype = 'processed_imgs'

        for i in range(x.shape[0]):
            folder_path = f'stimuli/{preprocessed_dir}/{dcnn_base}/{ftype}/task{stimulus_set}/{classes[i]}'
            if os.path.exists(folder_path) is False:
                os.makedirs(folder_path)

            # if save_as_reprs:
            # Save 1 image at a time, total 8 files per seed.
            np.save(os.path.join(folder_path, f'image{i}-{seed}.npy'), x[i])
            # # Only saving the original 8 once for each class.
            np.save(os.path.join(folder_path, f'image-orig{i}.npy'), x_orig[i])

            # TODO. For now we do not save image as .png but .npy 
            # because that is how gen.py is set up for `fulltrain`
            # elif save_as_imgs:
            # stimulus_set = config['stimulus_set']
            # img = Image.fromarray(np.rint(x[i]).astype('uint8'))
            # if not os.path.exists(f'stimuli/task{stimulus_set}_all/{classes[i]}/'):
            #     os.makedirs(f'stimuli/task{stimulus_set}_all/{classes[i]}/')
            # img.save(f'stimuli/task{stimulus_set}_all/{classes[i]}/image{i}-{seed}.png')
            # print(x[i])
            # plt.imshow(x[i]/255.)
            # plt.savefig(f'stimuli/task1_all/{classes[i]}/image{i}-{seed}.png')


def save_processed_data_54(model, config, preprocess_func):
    """
    Purpose:
    --------
    Due to big difference to task1-5, for task6 (Current Biology),
    we have a dedicated function for it. 
    
    We still produce layer representations and save them for stacking later. The difference
    here is the data augmentation applied:
        1. Different locations on the background.
        2. Zoom in/out.
        3. Add random uniform noise.
        4. Rotate a slight bit and do not get cut off.

    We also save the actual preprocessed images for visual examination.
    """
    classes = ['0000',
               '0001',
               '0010',
               '0011',
               '0100',
               '0101',
               '0110',
               '0111',
               '1000',
               '1001',
               '1010',
               '1011',
               '1100',
               '1101',
               '1110',
               '1111']

    path_raw = 'stimuli/task6_raw'        # from Mack, no background.
    image_path_parent = 'stimuli/task6_all_v3'   # save augmented images (bg added).

    preprocessed_dir = config['preprocessed_dir']
    dcnn_base = config['dcnn_base']
    layer = config['layer']
    stimulus_set = config['stimulus_set']
    reprs_path_parent = f'stimuli/{preprocessed_dir}/{dcnn_base}/{layer}_reprs/task{stimulus_set}'
    shrink_rates = np.linspace(
                        config['min_shrink_rate'], 
                        config['max_shrink_rate'], 
                        config['num_zooms'])

    np.random.seed(config['meta_seed'])
    # one stimulus type at a time.
    # each fname is like `0000.png`
    for fname in sorted(os.listdir(path_raw)):
        
        # make class-level directory one type at a time.
        # image_path is like `image_path/0000`
        image_path = os.path.join(image_path_parent, fname[:-4])
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        # reprs_path is like `reprs_path/0000`
        reprs_path = os.path.join(reprs_path_parent, fname[:-4])
        if not os.path.exists(reprs_path):
            os.makedirs(reprs_path)

        # one image at a time.
        fpath = os.path.join(path_raw, fname)
        for shrink_rate in shrink_rates:
            target_size = (int(300 * shrink_rate), int(700 * shrink_rate))

            # all possible locations
            center_coord = (112, 112)
            min_x = 0
            max_x = int(224 - target_size[1])
            min_y = 0
            max_y = int(224 - target_size[0])
            xs = np.linspace(min_x, max_x, config['num_locations'])
            ys = np.linspace(min_y, max_y, config['num_locations'])
            
            # each resized image will then go thru a few augmentations.
            resized_img = load_n_resize_img(fpath, target_size=target_size)
            
            # plot one image at a location.
            for loc in range(config['num_locations']):
                # offset is the coord of the upper left corner
                offset = (int(xs[loc]), int(ys[loc]))
                seed = np.random.randint(1, 1000)
                # at each location,
                # 1. we have neither rotated or noised
                # 2. we have noised but not rotated
                # 3. we have rotated but not noised
                # 4. we have both rotated and noised
                ct = 0
                for rotation_range in [None, config['rotation_range']]:
                    for noise_mode in [None, config['noise_mode']]:
                        img = augment_img(resized_img, offset, fname, 
                                         seed=seed,
                                         rotation_range=rotation_range, 
                                         noise_mode=noise_mode)

                        pasted_fname = f'image_{shrink_rate}_{loc}_{seed}_{ct}'
                        # save one pasted image at a time.
                        img.save(os.path.join(image_path, f'{pasted_fname}.png'))
                        ct += 1
                        # now we also want to save the fc1 reprs as input for training:
                        # first load in the saved img as array to be used by pretrained model:
                        x = img_to_array(img, data_format='channels_last')
                        # Pillow images should be closed after `load_img`,
                        # but not PIL images.
                        if hasattr(img, 'close'):
                            img.close()
                        x = preprocess_func(x)
                        x = np.expand_dims(x, axis=0)
                        layer_reprs = model.predict(x)
                        np.save(os.path.join(reprs_path, f'{pasted_fname}.npy'), layer_reprs)


if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config')
    args = parser.parse_args()

    config_version = f'config_{args.config}'
    config = load_config(config_version)
    execute(config)