import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   mask_zero=True)
        # self.lstm1 = tf.keras.layers.LSTM(self.enc_units,
        #                                  return_sequences=True,
        #                                  return_state=False,
        #                                  recurrent_initializer='glorot_uniform')
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=False,
                                         return_state=False,
                                         recurrent_initializer='glorot_uniform')
        # self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm)
        self.latent_mean = tf.keras.layers.Dense(self.enc_units)
        self.latent_logvar = tf.keras.layers.Dense(self.enc_units)
        # self.bn = tf.keras.layers.BatchNormalization()

    def __call__(self, x):
        x = self.embedding(x)
        # x = self.lstm1(x)
        # x = self.lstm2(x)
        output = self.lstm(x)
        mean = self.latent_mean(output)
        logvar = self.latent_logvar(output)
        latent = self._reparameterize(mean, logvar)
        # latent = output
        # mean, logvar = 0., 0.
        if self.training:
            return latent, mean, logvar
        else:
            return latent

    def _reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def initialize_hidden_state(self, batch_sz=None):
        if batch_sz is not None:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((batch_sz, self.enc_units))
        else:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self, batch_sz=None):
        if batch_sz is not None:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((batch_sz, self.enc_units))
        else:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((self.batch_sz, self.enc_units))

    def backward(self, loss, tape):
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        return gradients, variables

    def optimize(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def update(self, loss, tape):
        self.optimize(*self.backward(loss, tape))

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


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
