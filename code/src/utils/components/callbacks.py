import os

import tensorflow as tf

from typing import Union

# My Callback
class MyBackupAndRestore(tf.keras.callbacks.Callback):
    def __init__(self, backup_dir, optimizer, model, epoch_name: Union[str, None] = None, max_to_keep: Union[int, None] = None):
        super().__init__()
        self.epoch_name = epoch_name
        self._ckpt_saved_epoch = tf.Variable(
            initial_value=tf.constant(0, dtype=tf.int64), 
            name="ckpt_saved_epoch"
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, 
            model=model,
            ckpt_saved_epoch=self._ckpt_saved_epoch,
        )
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=backup_dir, 
            max_to_keep=max_to_keep)

    def on_epoch_begin(self, epoch, logs=None):
        self._ckpt_saved_epoch.assign(epoch + 1)
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # if self.epoch_name:
        #     self.manager._checkpoint_prefix = os.path.join(self.manager._directory, f"{self.epoch_name}-{epoch}/")
        self.manager.save()
        # save_path = checkpoint.save(checkpoint_directory)