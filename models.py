import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from layers import AttnFactory
from utils import load_config

try:
    from finetune.keras_custom import regularizers
except ModuleNotFoundError:
    from keras_custom import regularizers

"""
A database of all vision models
"""    

def model_base(
        config_version,
        input_shape=(224, 224, 3),
        intermediate_input=False,
    ):
    """
    Load DCNN model for finetuning. 
    The model has the option to insert attn layers at positions. 

    inputs:
    -------
        config_version: will load the following:
            model_name: vgg16 / vgg19 / resnet50
            layer: where to intercept representations.
            input_shape=(224, 224, 3),
            actv_func: prediction layer only, default `sigmoid`
            lr: default `3e-5`
            train: finetune / finetune-with-lowAttn
            stimulus_set: 1-6
            intermediate_input: whether to use intercepted layer as input.
                if False, we load the entire model regardless of `train`.

            if train == 'finetune-with-lowAttn':
                config will also load the following:
                attn_positions: ..
                attn_initializer: ..
                noise_level: ..
                noise_distribution: ..
                random_seed: ..
                low_attn_constraint: ..
                attn_regularizer: ..
                reg_strength: ..

    returns:
    --------
        model: based on `train`, the model returned can be 
            1. `train == 'none'`, return model intercepted at `layer`.
            2. `train == 'finetune*'` and `intermediate_input=True`,
                return entire model
            3. `train == 'finetune*'` and `intermediate_input=False`,
                return intercepted model.
        reprs_dims: the size of intercepted model (flattened)
        preprocess_func: model-specific preprocessing routine
    """
    config = load_config(config_version)
    model_name = config['model_name']
    layer = config['layer']
    actv_func = config['actv_func']
    lr = config['lr']
    train = config['train']
    stimulus_set = config['stimulus_set']

    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
        layer_reprs = model.get_layer(layer).output

    elif model_name == 'vgg19':
        model = tf.keras.applications.VGG19(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.vgg19.preprocess_input
        layer_reprs = model.get_layer(layer).output
    
    elif model_name == 'resnet50':
        model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.resnet.preprocess_input
        layer_reprs = model.get_layer(layer).output

    # if we intercept early, we need to flatten the conv block
    # and we the flattened reprs as inputs.
    if layer not in ['flatten', 'fc1', 'fc2']:
        layer_reprs = layers.Flatten()(layer_reprs)

    # --- return model based on whether fine tune ---
    if train == 'none':
        model = Model(inputs=model.input, outputs=layer_reprs)
        model.summary()
        reprs_dims = model.get_layer(layer).output.shape[-1]
        print(f'[Check] Cut off layer size:{reprs_dims}')
        return model, reprs_dims, preprocess_func

    # ------- stuff to do with the train task -------
    if stimulus_set not in ['6', 6]:
        pred_layer_units = 3
    else:
        pred_layer_units = 4

    # instantiate final layer
    pred_layer = layers.Dense(
        pred_layer_units, 
        activation=actv_func, 
        name='pred'
    )

    # Entire model (from input layer to pred layer)
    if intermediate_input is False:
        output = pred_layer(layer_reprs)
        model = Model(inputs=model.input, outputs=output)

    # Intercepted model (from target layer to pred layer)
    else:
        intermediate_input = layers.Input(shape=layer_reprs.shape[-1])
        output = pred_layer(intermediate_input)
        model = Model(inputs=intermediate_input, outputs=output)

    # Only finetune the output layer.
    if train == 'finetune':
        for layer in model.layers:
            if layer.name in ['pred']:
                continue
            else:
                layer.trainable = False

    # Also finetune attn layers.
    elif train == 'finetune-with-lowAttn':
        # attn layer positions
        attn_positions = config['attn_positions'].split(',')
        
        # attn layer settings
        if config['attn_initializer'] == 'ones':
            attn_initializer = tf.keras.initializers.Ones()
        elif config['attn_initializer'] == 'ones-withNoise':
            attn_initializer = initializers.NoisyOnes(
                noise_level=config['noise_level'], 
                noise_distribution=config['noise_distribution'], 
                random_seed=config['random_seed']
            )
                    
        if config['low_attn_constraint'] == 'nonneg':
            low_attn_constraint = tf.keras.constraints.NonNeg()
            
        if config['attn_regularizer'] == 'l1':
            attn_regularizer = tf.keras.regularizers.l1(config['reg_strength'])
        else:
            attn_regularizer = None

        # loop thru all layers and apply attn at positions.
        dcnn_layers = model.layers[1:]
        x = model.input
        fake_inputs = []
        for layer in dcnn_layers:

            # regardless of attn
            # apply one layer at a time from DCNN.
            layer.trainable = False
            x = layer(x)

            # apply attn at the output of the above layer output
            if layer.name in attn_positions:
                attn_size = x.shape[-1]

                fake_input = layers.Input(
                    shape=(attn_size,),
                    name=f'fake_input_{layer.name}'
                )
                fake_inputs.append(fake_input)
                
                attn_weights = AttnFactory(
                    output_dim=attn_size, 
                    input_shape=fake_input.shape,
                    name=f'attn_factory_{layer.name}',
                    initializer=attn_initializer,
                    constraint=low_attn_constraint,
                    regularizer=attn_regularizer
                )(fake_input)

                # reshape attn to be compatible.
                attn_weights = layers.Reshape(
                    target_shape=(1, 1, attn_weights.shape[-1]),
                    name=f'reshape_attn_{layer.name}')(attn_weights)

                # apply attn to prev layer output
                x = layers.Multiply(name=f'post_attn_actv_{layer.name}')([x, attn_weights])

    input_shape = model.input.shape[1:]

    if train == 'finetune-with-lowAttn':
        inputs = [model.inputs]
        inputs.extend(fake_inputs)
        model = Model(inputs=inputs, outputs=x, name='dcnn_model')

    model.compile(
        tf.keras.optimizers.Adam(lr=lr),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    return model, input_shape, preprocess_func


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

    model, _, _ = model_base(
        config_version='config_t1.vgg16.block4_pool.None.run1-with-lowAttn'
    )
    model.summary()
    plot_model(model, to_file='dcnn_with_lowAttn.png')
