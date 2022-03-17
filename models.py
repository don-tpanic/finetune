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

def LayerWise_AttnOp(x, layer, config):
    """
    Apply attn to a specific layer. 

    inputs:
    ------- 
        x: representation from the previous layer
        layer: layer that attn will be applied to
        config: provide attn settings
    
    return:
    -------
        x: post-attn representation of input x
        fake_input: corresponding fake ones in the size of the attn layer
    """
    if config['low_attn_initializer'] == 'ones':
        attn_initializer = tf.keras.initializers.Ones()
    elif config['low_attn_initializer'] == 'ones-withNoise':
        attn_initializer = initializers.NoisyOnes(
            noise_level=config['noise_level'], 
            noise_distribution=config['noise_distribution'], 
            random_seed=config['random_seed']
        )
                
    if config['low_attn_constraint'] == 'nonneg':
        low_attn_constraint = tf.keras.constraints.NonNeg()
        
    if config['low_attn_regularizer'] == 'l1':
        attn_regularizer = tf.keras.regularizers.l1(config['reg_strength'])
    else:
        attn_regularizer = None

    # -------------------------------------------------------------
    attn_size = x.shape[-1]
    fake_input = layers.Input(
        shape=(attn_size,),
        name=f'fake_input_{layer.name}'
    )
    
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
    return x, fake_input


def model_base(
        config,
        input_shape=(224, 224, 3),
        intermediate_input=False,
    ):
    """
    Load DCNN model for finetuning. 
    The model has the option to insert attn layers at positions. 

    inputs:
    -------
        config: will load the following:
            dcnn_base: vgg16 / vgg19 / resnet50

            layer: where to intercept representations 
            (NOTE the first attn layer will be applied to the output of this layer,
            if there is only one attn layer, it is the same as the last attn layer.)

            input_shape=(224, 224, 3),

            dcnn_actv_func: prediction layer only, default `sigmoid`

            lr_finetune: default `3e-5`

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
    dcnn_base = config['dcnn_base']
    config_version = config['config_version']

    if 'lowAttn' in config_version:
        # first attn layer will be applied after this layer 
        layer_begin = config['low_attn_positions'].split(',')[0]
        # final attn layer will be applied after this layer
        layer_end = config['low_attn_positions'].split(',')[-1]
    else:
        layer_begin = config['layer']
    
    actv_func = config['dcnn_actv_func']
    lr = config['lr_finetune']
    train = config['train']
    stimulus_set = config['stimulus_set']

    if dcnn_base == 'vgg16':
        model = tf.keras.applications.VGG16(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.vgg16.preprocess_input

    elif dcnn_base == 'vgg19':
        model = tf.keras.applications.VGG19(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.vgg19.preprocess_input
    
    elif dcnn_base == 'resnet50':
        model = tf.keras.applications.ResNet50(
            weights='imagenet', 
            include_top=True, 
            input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.resnet.preprocess_input


    # ----------------------------------------------------------------------
    # ---- build up models depending on scenarios ----
    input_shape = model.input.shape[1:]

    if stimulus_set not in ['6', 6]:
        pred_layer_units = 3
    else:
        pred_layer_units = 4

    # instantiate final layer
    PredLayer = layers.Dense(
        pred_layer_units, 
        activation=actv_func, 
        name='pred'
    )

    layer_begin_reprs = model.get_layer(layer_begin).output
    if layer_begin not in ['flatten', 'fc1', 'fc2']:
        original_layer_begin_shape = layer_begin_reprs.shape[1:]
        layer_begin_reprs = layers.Flatten()(layer_begin_reprs)

    # The model is used to produce data (X, Y)
    if train == 'none':
        model = Model(inputs=model.input, outputs=layer_begin_reprs)
        model.summary()
        reprs_dims = model.get_layer(layer_begin).output.shape[-1]
        print(f'[Check] Cut off layer size:{reprs_dims}')
        return model, reprs_dims, preprocess_func

    # NOTE: For backward compatibility
    # (where layer_begin directly connects to pred)
    elif train == 'finetune':

        if intermediate_input is False:
            output = PredLayer(layer_begin_reprs)  # flattened.
            model = Model(inputs=model.input, outputs=output)
        else:
            intermediate_input = layers.Input(
                shape=layer_begin_reprs.shape[-1]
            )

            # x = layers.Reshape(
            #     target_shape=original_layer_begin_shape
            # )(intermediate_input)
            # x = layers.Flatten()(x)
            # output = PredLayer(x)
            # NOTE: confirmed that Reshape-Flatten will not cause problem.

            output = PredLayer(intermediate_input)
            model = Model(inputs=intermediate_input, outputs=output)

    # ----------------------------------------------------------------------
    # New integration with attn layers.
    elif train == 'finetune-with-lowAttn':
        attn_positions = config['low_attn_positions'].split(',')
        dcnn_layers = model.layers[1:]
        fake_inputs = []

        if intermediate_input is False:
            x = model.input
            for layer in dcnn_layers:

                layer.trainable = False
                x = layer(x)

                # apply attn at the output of the above layer output
                if layer.name in attn_positions:
                    x, fake_input = LayerWise_AttnOp(x, layer, config)
                    fake_inputs.append(fake_input)

                    # The last attn layer will be connected to the final layer `PredLayer`
                    if layer.name == layer_end:
                        x = layers.Flatten()(x)
                        output = PredLayer(x)
                        break

            inputs = [model.inputs]

        else:
            intermediate_input = layers.Input(
                shape=layer_begin_reprs.shape[-1]
            )

            # NOTE(ken) hacky but needed due to initially data were saved as flattened.
            x = layers.Reshape(
                target_shape=original_layer_begin_shape
            )(intermediate_input)

            ignore_layer = True
            for layer in dcnn_layers:
                
                if layer.name != layer_begin and ignore_layer:
                    continue
                else:
                    ignore_layer = False  # permanently set flag for the rest.
                    layer.trainable = False

                    if layer.name != layer_begin:
                        x = layer(x)

                    if layer.name in attn_positions:
                        x, fake_input = LayerWise_AttnOp(x, layer, config)
                        fake_inputs.append(fake_input)
                        
                        # The last attn layer will be connected to the final layer `PredLayer`
                        if layer.name == layer_end:
                            x = layers.Flatten()(x)
                            output = PredLayer(x)
                            break

            inputs = [intermediate_input]

        inputs.extend(fake_inputs)
        model = Model(inputs=inputs, outputs=output, name='dcnn_model')

    model.compile(
        tf.keras.optimizers.Adam(lr=lr),
        loss=tf.keras.losses.BinaryCrossentropy()
    )
    return model, input_shape, preprocess_func


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "-1"
    config = load_config(
        'config_t1.vgg16.block4_pool.None.run1-with-lowAttn'
    )
    model, _, _ = model_base(
        config,
        intermediate_input=True
    )
    model.summary()
    plot_model(model, to_file='dcnn_model.png')
