import numpy as np
import tensorflow as tf


class PredictionMonitor(tf.keras.callbacks.Callback):

    def __init__(self, config):
        super(PredictionMonitor, self).__init__()
        self.config = config
        self.log = []

    def on_epoch_end(self, epoch, logs=None):
        
        preprocessed_dir = self.config['preprocessed_dir']
        model_name = self.config['model_name']
        layer = self.config['layer']
        stimulus_set = self.config['stimulus_set']
        heldout = self.config['heldout']
        config_version = self.config['config_version']

        if stimulus_set not in [6, '6']:
            all_classes = ['000', '001', '010', '011',
                        '100', '101', '110', '111']
        else:
            NotImplementedError('Not support monitoring task6.')
            
        idx = 0
        for i in range(len(all_classes)):
            if all_classes[i] == heldout:
                idx = i 
                break

        data_path = f'stimuli/{preprocessed_dir}/{model_name}/{layer}_reprs/' \
                    f'task{stimulus_set}/{heldout}/image-orig{idx}.npy'
        # predict on the original stimulus (heldout)
        x = np.load(data_path)
        x = tf.expand_dims(x, axis=0)
        y = self.model.predict(x)
        self.log.extend(y)
        np.save(f'results/{config_version}/prediction_monitor.npy', self.log)
        

        


        
        




