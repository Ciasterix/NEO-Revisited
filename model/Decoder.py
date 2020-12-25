import tensorflow as tf

# from model.Attention import Attention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # self.lstm = tf.keras.layers.LSTM(self.dec_units,
        #                                  return_sequences=True,
        #                                  return_state=True,
        #                                  recurrent_initializer='glorot_uniform')
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        # self.attention = Attention()
        # self.attention = tf.keras.layers.Attention()
        self.attention = tf.keras.layers.Attention(dropout=0.0, use_scale=True)
        # E_proj = get_EF(input_size=40, dim=16, method="learnable", head_dim=1)
        # F_proj = get_EF(input_size=40, dim=16, method="learnable", head_dim=1)
        # self.attention = LinAttention(dim=128, dropout=0.0, E_proj=E_proj, F_proj=F_proj, full_attention=False)

    def __call__(self, x, hidden_state_in, enc_output, mask=None):
        # enc_output shape == (batch_size, max_length, hidden_size)
        hidden_state = tf.expand_dims(hidden_state_in, axis=1)
        # print("dec in", hidden_state.shape, enc_output.shape)
        if mask is not None:
            context_vector = self.attention(inputs=[hidden_state, enc_output],
                                            mask=[None, mask],
                                            training=self.training)
        else:
            context_vector = self.attention(inputs=[hidden_state, enc_output],
                                            training=self.training)
        # context_vector = self.attention(Q=hidden_state, K=enc_output, V=enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # print("x", x.shape)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # context_vector = tf.keras.layers.Dropout(0.9)(context_vector, training=self.training)
        x = tf.concat([context_vector, x], axis=-1)
        # x = context_vector

        # print("after concat", x.shape)

        # passing the concatenated vector to the GRU
        # output, hidden_state, cell_state = self.lstm(x)
        # print(hidden_state.shape)
        output, hidden_state = self.gru(x, initial_state=hidden_state_in)
        # output, hidden_state = self.gru2(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # print("output", output.shape)

        # output shape == (batch_size, vocab)
        x = self.fc(output)
        # print("x", x.shape)

        return x, hidden_state, None, None  # attention_weights

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
