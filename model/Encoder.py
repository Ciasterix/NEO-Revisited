import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.optimizer = tf.keras.optimizers.Adam()

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, hidden_state, cell_state = self.lstm(x, initial_state=hidden)
        return output, hidden_state, cell_state

    def initialize_hidden_state(self, batch_sz=None):
        if batch_sz is not None:
            return tf.zeros((batch_sz, self.enc_units))
        else:
            return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self, batch_sz=None):
        if batch_sz is not None:
            return tf.zeros((batch_sz, self.enc_units))
        else:
            return tf.zeros((self.batch_sz, self.enc_units))

    def backward(self, loss, tape):
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        return gradients, variables

    def optimize(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def update(self, loss, tape):
        self.optimize(*self.backward(loss, tape))

if __name__ == "__main__":
    BATCH_SIZE = 256
    vocab_inp_size = 32
    vocab_tar_size = 32
    embedding_dim = 64
    units = 128

    # Encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    example_input_batch = tf.random.uniform(shape=(BATCH_SIZE, 16), minval=0,
                                            maxval=31,
                                            dtype=tf.int64)
    example_target_batch = tf.random.uniform(shape=(BATCH_SIZE, 11), minval=0,
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
        sample_cell.shape))
