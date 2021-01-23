import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, dec_units,
                 batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # self.embedding = tf.keras.layers.Embedding(vocab_inp_size,
        #                                            embedding_dim,
        #                                            mask_zero=True)
        # self.lstm1 = tf.keras.layers.LSTM(self.dec_units,
        #                                  return_sequences=True,
        #                                  return_state=False,
        #                                  recurrent_initializer='glorot_uniform')
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=False,
                                         recurrent_initializer='glorot_uniform')

        # self.latent_to_hidden = tf.keras.layers.Dense(self.dec_units, activation="tanh")
        # self.bn = tf.keras.layers.BatchNormalization()

        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_tar_size))

    def __call__(self, latent, max_size):
        # mask = tf.cast(x, dtype=bool)
        # latent = tf.expand_dims(latent, axis=1)
        latent = tf.keras.layers.RepeatVector(max_size)(latent)
        output = self.lstm(latent)
        # output = output * tf.cast(mask, dtype=tf.float32)
        # hidden_state = tf.where(mask, hidden_state, states[0])
        # cell_state = tf.where(mask, cell_state, states[1])
        # states = [hidden_state, cell_state]
        x = self.out(output)
        return x

    def initialize_hidden_state(self, batch_sz=None):
        if batch_sz is not None:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((batch_sz, self.dec_units))
        else:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((self.batch_sz, self.dec_units))

    def initialize_cell_state(self, batch_sz=None):
        if batch_sz is not None:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((batch_sz, self.dec_units))
        else:
            # return tf.random.normal(shape=[batch_sz, self.enc_units])
            return tf.zeros((self.batch_sz, self.dec_units))

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
