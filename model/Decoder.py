import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_inp_size,
                                                   embedding_dim,
                                                   mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)

        self.att = tf.keras.layers.Dense(self.dec_units,
                                         activation="tanh",
                                         use_bias=False)
        self.out = tf.keras.layers.Dense(vocab_tar_size,
                                         activation="softmax",
                                         use_bias=False)

        self.optimizer = tf.keras.optimizers.Adam()
        self.attention = tf.keras.layers.Attention()

    def __call__(self, x, context_vector, enc_output, states, enc_mask=None):
        # hidden_state, cell_state = states
        x = self.embedding(x)
        mask = x._keras_mask
        x = tf.concat([x, context_vector], axis=-1)
        output, hidden_state, cell_state = self.lstm(x, initial_state=states)
        hidden_state = tf.where(mask, hidden_state, states[0])
        cell_state = tf.where(mask, cell_state, states[1])
        output = output * tf.expand_dims(
            tf.cast(mask, dtype=tf.float32), axis=2)

        # Attention
        x = tf.expand_dims(hidden_state, axis=1)
        if enc_mask is None:
            enc_mask = enc_output._keras_mask
        context_vector = self.attention(inputs=[x, enc_output],
                                        mask=[mask, enc_mask],
                                        training=self.training)
        x = self.concat2([output, context_vector])
        output = tf.reshape(x, (-1, x.shape[2]))
        context_shape = context_vector.shape
        context_vector = tf.reshape(self.att(output), context_shape)

        output = tf.keras.layers.Reshape(
            (context_vector.shape[2],))(context_vector)
        x = self.out(output)

        return x, context_vector, [hidden_state, cell_state], None

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
