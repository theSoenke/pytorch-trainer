
import os
import shutil

import numpy as np


class ModelCheckpoint():
    def __init__(self, directory, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='max', prefix=''):
        super().__init__()
        self.monitor = monitor
        self.directory = directory
        self.save_best_only = save_best_only
        self.prefix = prefix
        self.last_checkpoint_path = None

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def save_model(self, filepath, save_func, overwrite):
        dirpath = '/'.join(filepath.split('/')[:-1])
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if overwrite:
            if self.last_checkpoint_path != None:
                os.remove(self.last_checkpoint_path)

        self.last_checkpoint_path = filepath
        save_func(filepath)

    def on_epoch_end(self, epoch, save_func, logs=None):
        logs = logs or {}
        filepath = '{}/{}_ckpt_epoch_{}.ckpt'.format(self.directory, self.prefix, epoch + 1)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                print('Can save best model only with %s available,' ' skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' ' saving model to %s' %
                              (epoch + 1, self.monitor, self.best, current, filepath))
                    self.best = current
                    self.save_model(filepath, save_func, overwrite=True)

                else:
                    print('\nEpoch %05d: %s did not improve' % (epoch + 1, self.monitor))
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            self.save_model(filepath, save_func, overwrite=False)
