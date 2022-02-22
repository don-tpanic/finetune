import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.keras import activations


class AttnFactory(Layer):
    """
    Given some fake inputs (from a second input layer) that are fixed
    values, this layer maps the inputs to a set of attention weights.
    The mapping is element-wise multiplication (not dot product). 
    The actual trainable layers are the connections between the inputs 
    and the output units that return attention weights.

    inputs:
    -------
        fake inputs that are fixed 

    operation:
    ----------
        elementwise multiplication

    outputs:
    --------
        Filter-wise attention weights
    """
    def __init__(
        self, 
        output_dim, 
        initializer, 
        constraint,
        regularizer,
        **kwargs):

        self.output_dim = output_dim
        self.initializer = initializer
        self.constraint = constraint
        self.regularizer = regularizer

        super(AttnFactory, self).__init__(**kwargs)

    def build(self, input_shape):
        # NOTE(ken) No bias terms for now.
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.output_dim,),
            initializer=self.initializer,
            constraint=self.constraint,
            regularizer=self.regularizer,
            trainable=True)
        super(AttnFactory, self).build(input_shape)

    def call(self, x):
        """
        Elementwise multiply.
        """
        return gen_math_ops.mul(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(AttnFactory, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config
