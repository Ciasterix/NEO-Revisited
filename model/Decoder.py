import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_inp_size, embedding_dim, mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)
        self.concat2 = tf.keras.layers.Concatenate(axis=-1)
        self.fc = tf.keras.layers.Dense(vocab_tar_size)

        self.optimizer = tf.keras.optimizers.Adam()

        self.attention = tf.keras.layers.Attention()

    def __call__(self, x, context_vector, enc_output, states):
        hidden_state, cell_state = states
        x = self.embedding(x)
        mask = x._keras_mask
        x = tf.concat([x, context_vector], axis=-1)

        output, hidden_state, cell_state = self.lstm(x, initial_state=states)
        hidden_state = tf.where(mask, hidden_state, states[0])
        cell_state = tf.where(mask, cell_state, states[1])
        output = output*tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=2)

        # Attention
        x = tf.expand_dims(hidden_state, axis=1)
        context_vector = self.attention(inputs=[x, enc_output],
                                        mask=[mask, enc_output._keras_mask])

        x = self.concat2([output, context_vector])
        output = tf.keras.layers.Reshape((x.shape[2],))(x)
        x = self.fc(output)

        return x, context_vector, [hidden_state, cell_state], None#attention_weights

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
