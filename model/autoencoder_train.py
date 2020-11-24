import time

import tensorflow as tf

from model.Dataset import Dataset
from model.Decoder import Decoder
from model.Encoder import Encoder


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def main():
    pass


if __name__ == "__main__":
    BATCH_SIZE = 64
    vocab_inp_size = 32
    vocab_tar_size = 32
    embedding_dim = 64
    units = 128

    # Encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    # Decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


    @tf.function
    def train_step(inp, targ, enc_hidden, enc_cell):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = encoder(inp,
                                                       [enc_hidden, enc_cell])

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims(
                [0] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _, _ = decoder(dec_input, dec_hidden,
                                                        enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


    EPOCHS = 5
    steps_per_epoch = 5
    dataset = Dataset()

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        enc_cell = encoder.initialize_cell_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, enc_cell)
            total_loss += batch_loss

            if batch % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        # if (epoch + 1) % 2 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # Test prediction
    BATCH_SIZE = 1
    test_input_batch = tf.random.uniform(shape=(BATCH_SIZE, 16), minval=0,
                                         maxval=2,
                                         dtype=tf.int64)
    enc_hidden = encoder.initialize_hidden_state(batch_sz=BATCH_SIZE)
    enc_cell = encoder.initialize_cell_state(batch_sz=BATCH_SIZE)
    print(test_input_batch.numpy())
    enc_output, enc_hidden, _ = encoder(test_input_batch,
                                        [enc_hidden, enc_cell])
    dec_input = tf.expand_dims(
        [0] * BATCH_SIZE, 1)
    dec_hidden = enc_hidden
    print("pred:", dec_input.shape, dec_hidden.shape, enc_output.shape)
    test_output_batch, _, _, _ = decoder(dec_input, dec_hidden, enc_output)
    test_output_batch = tf.argmax(test_output_batch[0]).numpy()
    print("Predicted output:", test_output_batch)
    print("Ground truth output:", test_input_batch[:BATCH_SIZE, :11].numpy())
