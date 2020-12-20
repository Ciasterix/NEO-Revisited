import tensorflow as tf


# from model.Attention import Attention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        # self.attention = Attention()
        self.attention = tf.keras.layers.Attention(dropout=0.5, use_scale=True)

    def __call__(self, x, hidden_state, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        hidden_state = tf.expand_dims(hidden_state, axis=1)
        # print("dec in", hidden_state.shape, enc_output.shape)
        context_vector = self.attention(inputs=[hidden_state, enc_output],
                                        training=self.training)
        # print("context", context_vector.shape)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # print("x", x.shape)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x = tf.concat([context_vector, x], axis=-1)
        # print("after concat", x.shape)

        # passing the concatenated vector to the GRU
        output, hidden_state, cell_state = self.lstm(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # print("output", output.shape)

        # output shape == (batch_size, vocab)
        x = self.fc(output)
        # print("x", x.shape)

        return x, hidden_state, cell_state, None  # attention_weights

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
