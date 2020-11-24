import tensorflow as tf

from model.Encoder import Encoder


class Surrogate(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(Surrogate, self).__init__()
        self.fc = tf.keras.layers.Dense(hidden_size)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.fc(inputs)
        x = self.out(x)
        return x


if __name__ == "__main__":
    BATCH_SIZE = 64
    vocab_inp_size = 32
    vocab_tar_size = 32
    embedding_dim = 64
    units = 128

    # Encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    example_input_batch = tf.random.uniform(shape=(64, 16), minval=0, maxval=31,
                                            dtype=tf.int64)
    example_target_batch = tf.random.uniform(shape=(64, 11), minval=0,
                                             maxval=31, dtype=tf.int64)
    print(example_input_batch.shape, example_target_batch.shape)
    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    sample_cell = encoder.initialize_cell_state()
    sample_output, sample_hidden, sample_cell = encoder(example_input_batch,
                                                        [sample_hidden,
                                                         sample_cell])
    print(
        'Encoder output shape: (batch size, sequence length, units) {}'.format(
            sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(
        sample_hidden.shape))
    print('Encoder Cell state shape: (batch size, units) {}'.format(
        sample_hidden.shape))
    # Surrogate
    surrogate = Surrogate(hidden_size=128)
    surrogate_output = surrogate(sample_hidden)
    print(surrogate_output.shape)