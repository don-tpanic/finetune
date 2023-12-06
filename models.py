import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Flatten

try:
    from finetune.keras_custom import regularizers
except ModuleNotFoundError:
    from keras_custom import regularizers

"""
A database of all vision models
"""    

def model_base(model_name, 
                layer='fc2',
                input_shape=(224, 224, 3),
                actv_func='sigmoid',
                kernel_constraint=None,
                kernel_regularizer=None,
                hyperbolic_strength=None,
                lr=3e-5,
                train='finetune',
                stimulus_set=6,
                intermediate_input=False):
    """
    Model constructor.

    inputs:
    -------
        model_name: vgg16 / vgg19 / vit_b16
        layer: where to intercept representations.
        input_shape=(224, 224, 3),
        actv_func: prediction layer only, default `sigmoid`
        kernel_constraint: prediction layer only, default None
        hyperbolic_strength: prediction layer only, default None
        lr: default 3e-5
        train: finetune / fulltrain
        stimulus_set: 1-6
        intermediate_input: whether to use intercepted layer as input.
            if False, we load the entire model regardless of `train`.

    returns:
    --------
        model: based on `train`, the model returned can be 
            1. `train == 'none'`, return model intercepted at `layer`.
            2. `train == 'finetune | fulltrain` and `intermediate_input=True`,
                return entire model
            3. `train == 'finetune | fulltrain` and `intermediate_input=False`,
                return intercepted model.
        reprs_dims: the size of intercepted model (flattened)
        preprocess_func: model-specific preprocessing routine
    """
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(
                weights='imagenet', include_top=True, input_shape=input_shape)
        preprocess_func = tf.keras.applications.vgg16.preprocess_input
        layer_reprs = model.get_layer(layer).output

    elif model_name == 'vgg19':
        model = tf.keras.applications.VGG19(
                weights='imagenet', include_top=True, input_shape=input_shape)
        preprocess_func = tf.keras.applications.vgg19.preprocess_input
        layer_reprs = model.get_layer(layer).output
    
    elif model_name == 'resnet50':
        model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=True, input_shape=input_shape
        )
        preprocess_func = tf.keras.applications.resnet.preprocess_input
        layer_reprs = model.get_layer(layer).output
    
    elif model_name == 'vit_b16':
        from transformers_custom import AutoImageProcessor, TFViTModel
        model = TFViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            cache_dir='model_zoo/vit_b16'
        )
        preprocess_func = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            cache_dir='model_zoo/vit_b16'
        )
    
    if 'vit' not in model_name:
        # if we intercept early, we need to flatten the conv block
        # and we the flattened reprs as inputs.
        if layer not in ['flatten', 'fc1', 'fc2']:
            layer_reprs = Flatten()(layer_reprs)

        # --- return model based on whether fine tune ---
        if train == 'none':
            model = Model(inputs=model.input, outputs=layer_reprs)
            model.summary()
            reprs_dims = model.get_layer(layer).output.shape[-1]
            print(f'[Check] Cut off layer size:{reprs_dims}')
            return model, reprs_dims, preprocess_func

        # ------- stuff to do with the train task -------
        if hyperbolic_strength is None:
            actv_regularizer = None
        else:
            actv_regularizer = regularizers.hyperbolic(strength=hyperbolic_strength)

        if kernel_constraint == 'nonneg':
            kernel_constraint = tf.keras.constraints.NonNeg()
        else:
            kernel_constraint = None
        if kernel_regularizer == 'l1':
            kernel_regularizer = tf.keras.regularizers.l1(1e-2)
        elif kernel_regularizer == 'l2':
            kernel_regularizer = tf.keras.regularizers.l2(1e-2)
        else:
            kernel_regularizer = None

        if stimulus_set not in ['6', 6]:
            pred_layer_units = 3
        else:
            pred_layer_units = 4

        # instantiate final layer
        pred_layer = Dense(pred_layer_units, 
                        activation=actv_func, 
                        kernel_constraint=kernel_constraint,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=actv_regularizer,
                        name='pred')

        # Entire model (from input layer to pred layer)
        if intermediate_input is False:
            output = pred_layer(layer_reprs)
            model = Model(inputs=model.input, outputs=output)
        # Intercepted model (from target layer to pred layer)
        else:
            intermediate_input = Input(shape=layer_reprs.shape[-1])
            output = pred_layer(intermediate_input)
            model = Model(inputs=intermediate_input, outputs=output)

        if train == 'finetune':
            for layer in model.layers:
                if layer.name in ['pred']:
                    continue
                else:
                    layer.trainable = False

        model.compile(tf.keras.optimizers.Adam(lr=lr),
                    loss=tf.keras.losses.BinaryCrossentropy())

        # Add metric monitoring layer activity
        if hyperbolic_strength is not None:
            hyperbolic_loss = tf.reduce_sum(
                -hyperbolic_strength * tf.square(output) + hyperbolic_strength * output)
            model.add_metric(hyperbolic_loss, name='hyperbolic_loss')
        input_shape = model.input.shape[1:]

    # =============== For ViT only ===============
    else:
        if train == "none":
            # We defer layer interception to
            # data.py (i.e., we do not build a model here)
            # Change to channel first for ViT HF's implementation
            input_shape = input_shape[::-1] # (224, 224, 3) -> (3, 224, 224)

        elif train == "finetune":
            # For ViT, only use the `intermediate_input` option.
            # So we just need to get the layer output and finetune
            # the final Pred layer.
            # ------- stuff to do with the train task -------
            if hyperbolic_strength is None:
                actv_regularizer = None
            else:
                actv_regularizer = regularizers.hyperbolic(strength=hyperbolic_strength)

            if kernel_constraint == 'nonneg':
                kernel_constraint = tf.keras.constraints.NonNeg()
            else:
                kernel_constraint = None
            if kernel_regularizer == 'l1':
                kernel_regularizer = tf.keras.regularizers.l1(1e-2)
            elif kernel_regularizer == 'l2':
                kernel_regularizer = tf.keras.regularizers.l2(1e-2)
            else:
                kernel_regularizer = None

            if stimulus_set not in ['6', 6]:
                pred_layer_units = 3
            else:
                pred_layer_units = 4

            # instantiate final layer
            pred_layer = Dense(pred_layer_units, 
                            activation=actv_func, 
                            kernel_constraint=kernel_constraint,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=actv_regularizer,
                            name='pred')

            # Entire model (from input layer to pred layer)
            if intermediate_input is False:
                output = pred_layer(layer_reprs)
                model = Model(inputs=model.input, outputs=output)
            # Intercepted model (from target layer to pred layer)
            else:
                # HACK: use a synthetic input to get the layer reprs
                synthetic_input = tf.random.uniform(
                    (1, 3, 224, 224), minval=0, maxval=1
                )
                layer_index = int(layer[6:7])  # `layer_x_..`
                if 'msa' in layer:
                    # Grabs the MSA outputs 
                    # \in  (bs, seq_len, num_heads, head_dim)
                    # e.g. (1,  197,    12,         64)
                    layer_reprs = model(
                        synthetic_input, training=False, 
                        output_msa_states=True
                    ).attentions[layer_index].numpy()
                else:
                    layer_reprs = model(
                        synthetic_input, training=False, 
                        output_hidden_states=True
                    ).hidden_states[layer_index].numpy()

                # Flatten the non-batch dimensions
                layer_reprs = layer_reprs.reshape(
                    layer_reprs.shape[0], -1
                )
                
                intermediate_input = Input(shape=layer_reprs.shape[-1])
                output = pred_layer(intermediate_input)
                model = Model(inputs=intermediate_input, outputs=output)

            if train == 'finetune':
                for layer in model.layers:
                    if layer.name in ['pred']:
                        continue
                    else:
                        layer.trainable = False

            model.compile(tf.keras.optimizers.Adam(lr=lr),
                        loss=tf.keras.losses.BinaryCrossentropy())

            # Add metric monitoring layer activity
            if hyperbolic_strength is not None:
                hyperbolic_loss = tf.reduce_sum(
                    -hyperbolic_strength * tf.square(output) + hyperbolic_strength * output)
                model.add_metric(hyperbolic_loss, name='hyperbolic_loss')
            input_shape = model.input.shape[1:]
            
    return model, input_shape, preprocess_func


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

    model, _, _ = model_base(
              model_name='vit_b16',
              layer='layer_3',
              train='finetune',
              stimulus_set=1,
              intermediate_input=True
              )
    model.summary()
