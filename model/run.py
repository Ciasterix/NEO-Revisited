import tensorflow as tf

from model.Attention import Attention
from model.Decoder import Decoder
from model.Encoder import Encoder

if __name__ == "__main__":
    BATCH_SIZE = 64
    vocab_inp_size = 32
    vocab_tar_size = 32
    embedding_dim = 256
    units = 1024

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
    sample_output, sample_hidden, cell_hidden = encoder(example_input_batch, [sample_hidden, sample_cell])
    print(
        'Encoder output shape: (batch size, sequence length, units) {}'.format(
            sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(
        sample_hidden.shape))
    print('Encoder Cell state shape: (batch size, units) {}'.format(
        sample_hidden.shape))

    # Attention
    attention_layer = Attention()
    attention_result, attention_weights = attention_layer(sample_hidden,
                                                          sample_output)

    print("Attention result shape: (batch size, units) {}".format(
        attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(
        attention_weights.shape))

    # Decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    sample_decoder_output, _, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                          sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(
        sample_decoder_output.shape))
