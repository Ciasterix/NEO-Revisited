import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, dec_units,
                 batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_inp_size,
                                                   embedding_dim,
                                                   mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=False,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

        self.latent_to_hidden = tf.keras.layers.Dense(self.dec_units, activation="tanh")
        self.bn = tf.keras.layers.BatchNormalization()

        self.out = tf.keras.layers.Dense(vocab_tar_size,
                                         activation="softmax",
                                         use_bias=False)

        self.optimizer = tf.keras.optimizers.Adam()
        self.attention = tf.keras.layers.Attention()

    def __call__(self, x, states):
        hidden_state, cell_state = states
        hidden_state = self.latent_to_hidden(hidden_state)
        hidden_state = self.bn(hidden_state, training=self.training)
        states = [hidden_state, cell_state]
        x = self.embedding(x)
        mask = x._keras_mask
        x._keras_mask = None

        output, hidden_state, cell_state = self.lstm(x, initial_state=states)
        output = output * tf.cast(mask, dtype=tf.float32)
        hidden_state = tf.where(mask, hidden_state, states[0])
        cell_state = tf.where(mask, cell_state, states[1])

        x = self.out(output)

        return x, [hidden_state, cell_state]

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
