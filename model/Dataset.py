import tensorflow as tf


class Dataset:
    def __init__(self):
        pass

    def __call__(self, steps_per_epoch):
        i = 0
        while i < steps_per_epoch:
            i += 1
            example_input_batch = tf.random.uniform(shape=(64, 16), minval=0,
                                                    maxval=31,
                                                    dtype=tf.int64)
            example_target_batch = tf.random.uniform(shape=(64, 11), minval=0,
                                                     maxval=31, dtype=tf.int64)
            yield example_input_batch, example_target_batch
