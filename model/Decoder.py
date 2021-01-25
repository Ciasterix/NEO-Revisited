import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, dec_units,
                 batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding_dim = embedding_dim
        self.vocab_tar_size = vocab_tar_size
        self.embedding = tf.keras.layers.Embedding(vocab_inp_size,
                                                   embedding_dim)
        # self.lstm1 = tf.keras.layers.LSTM(self.dec_units,
        #                                  return_sequences=True,
        #                                  return_state=False,
        #                                  recurrent_initializer='glorot_uniform')
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=False,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        # self.bi_lstm = tf.keras.layers.Bidirectional(self.lstm)

        # self.latent_to_hidden = tf.keras.layers.Dense(self.dec_units, activation="tanh")
        # self.bn = tf.keras.layers.BatchNormalization()

        self.out = tf.keras.layers.Dense(vocab_tar_size)

    def __call__(self, token, latent, states):
        mask = tf.cast(token, dtype=bool)

        x = self.embedding(token)
        x = tf.concat([x, latent], axis=2)
        x, h, c = self.lstm(x, initial_state=states)
        x, h, c = self.apply_masking(x, h, c, mask, states)
        states = [h, c]
        x = self.out(x)

        return x, states

    def apply_masking(self, output, h, c, mask, states):
        output = output * tf.cast(mask, dtype=tf.float32)
        h = tf.where(mask, h, states[0])
        c = tf.where(mask, c, states[1])
        return output, h, c

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
