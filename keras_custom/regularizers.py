import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.keras.regularizers import Regularizer 
from tensorflow.keras import backend as K


class Hyperbolic(Regularizer):
    """
    Purpose:
    --------
        Hyperbolic regularisation that pushes layer activations
        to be as close as 0 or 1.
    """
    def __init__(self, strength):
        print(f'[Check] initialised Hyperbolic class')
        self.strength = strength

    def __call__(self, x):
        # y = ax^2 + bx
        # a = - strength, b = strength
        b = self.strength
        a = -b
        return tf.reduce_sum(a * tf.square(x) + b * x)

    def get_config(self):
        return {'strength': self.strength}

def hyperbolic(strength):
    print(f'[Check] calling hyperbolic func')
    return Hyperbolic(strength)