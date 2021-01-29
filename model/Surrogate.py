import tensorflow as tf

from model.Attention import Attention
from model.Encoder import Encoder


class Surrogate(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Surrogate, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(hidden_size)
        # self.fc1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        # self.fc2 = tf.keras.layers.Dense(hidden_size, activation="relu")
        # self.fc3 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.out = tf.keras.layers.Dense(64, activation="sigmoid")

        self.attention = Attention()

        self.optimizer = tf.keras.optimizers.Adam()

    def __call__(self, hidden_state):
        # x = self.fc1(hidden_state)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.lstm1(hidden_state)
        x = self.lstm2(x)
        x = self.out(x)
        return x

    def backward(self, loss, tape):
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        return gradients, variables

    def optimize(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def update(self, loss, tape):
        self.optimize(*self.backward(loss, tape))


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
