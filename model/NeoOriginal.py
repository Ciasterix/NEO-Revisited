import time

import tensorflow as tf

from model.Dataset import Dataset
from model.Encoder import Encoder
from model.Decoder import Decoder


class NeoOriginal:

    def __init__(  # TODO move parameters to config file
            self,
            batch_size=64,
            vocab_inp_size=32,
            vocab_tar_size=32,
            embedding_dim=64,
            units=128,
            epochs=200,
            epoch_decay=1,
            min_epochs=10,
            steps_per_epoch=5,
            verbose=True
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.min_epochs = min_epochs
        self.steps_per_epoch = steps_per_epoch

        self.verbose = verbose

        self.enc = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        self.dec = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        self.dataset = Dataset()

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    @tf.function
    def train_step(self, inp, targ, enc_hidden, enc_cell):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = self.enc(
                inp, [enc_hidden, enc_cell])

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([0] * self.batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _, _ = self.dec(
                    dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.enc.trainable_variables + self.dec.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def __train_autoencoder(self):
        dataset = Dataset()

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.enc.initialize_hidden_state()
            enc_cell = self.enc.initialize_cell_state()
            total_loss = 0

            data_generator = dataset(self.steps_per_epoch)

            for (batch, (inp, targ)) in enumerate(data_generator):
                batch_loss = self.train_step(inp, targ, enc_hidden, enc_cell)
                total_loss += batch_loss

                if batch % 1 == 0 and self.verbose:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy()))

            if self.verbose:
                print('Epoch {} Loss {:.4f}'.format(
                    epoch + 1, total_loss / self.steps_per_epoch))
                print('Time for epoch {} sec\n'.format(time.time() - start))

        # decrease number of epoch, but don't go below self.min_epochs
        self.epochs = max(self.epochs - self.epoch_decay, self.min_epochs)

    def breed(self):
        self.__train_autoencoder()


if __name__ == "__main__":
    neo = NeoOriginal(epochs=15)
    neo.breed()
    neo.breed()  # second call to check epoch decay
