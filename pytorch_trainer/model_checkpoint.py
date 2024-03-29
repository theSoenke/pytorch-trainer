
import os
import shutil

import numpy as np


class ModelCheckpoint():
    def __init__(self, directory, monitor='val_loss', save_best_only=False, period=1, save_weights_only=False, mode='max', prefix=''):
        super().__init__()
        self.monitor = monitor
        self.directory = directory
        self.save_best_only = save_best_only
        self.prefix = prefix
        self.last_checkpoint_path = None
        self.period = period
        self.epochs_since_saved = 0

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

    def on_epoch_end(self, epoch, save_func, seed, logs=None):
        self.epochs_since_saved += 1
        if self.epochs_since_saved < self.period:
            return

        filepath = f"{self.directory}/{self.prefix}_ckpt_epoch_{epoch}_seed_{seed}.ckpt"
        if self.save_best_only:
            logs = logs or {}
            current = logs.get(self.monitor)
            if current is None:
                print(f"Can save best model only with {self.monitor} available\n")
            else:
                if self.monitor_op(current, self.best):
                    print(f"Epoch {epoch:05d}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {filepath}\n")
                    self.best = current
                    self.save_model(filepath, save_func, overwrite=True)

                else:
                    print(f"Epoch {epoch:05d}: {self.monitor} did not improve\n")
        else:
            print(f"Epoch {epoch:05d}: saving model to {filepath}\n")
            self.save_model(filepath, save_func, overwrite=False)

        self.epochs_since_saved = 0
