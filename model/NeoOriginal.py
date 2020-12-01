import time

import tensorflow as tf

from model.Dataset import Dataset
from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Surrogate import Surrogate


class NeoOriginal:

    def __init__(  # TODO move parameters to config file
            self,
            batch_size=64,
            vocab_inp_size=32,
            vocab_tar_size=32,
            embedding_dim=64,
            units=128,
            hidden_size=128,
            alpha=0.1,
            epochs=200,
            epoch_decay=1,
            min_epochs=10,
            steps_per_epoch=5,
            verbose=True
    ):
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.min_epochs = min_epochs
        self.steps_per_epoch = steps_per_epoch

        self.verbose = verbose

        self.enc = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        self.dec = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        self.surrogate = Surrogate(hidden_size)
        self.dataset = Dataset()

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    @tf.function
    def train_step(self, inp, targ, targ_surrogate, enc_hidden,
                   enc_cell):
        autoencoder_loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = self.enc(
                inp, [enc_hidden, enc_cell])

            surrogate_output = self.surrogate(enc_hidden)
            surrogate_loss = self.surrogate_loss_function(targ_surrogate,
                                                          surrogate_output)

            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([0] * self.batch_size,
                                       1)  # TODO: Change [0] to start token
            print(dec_input.shape)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _, _ = self.dec(
                    dec_input, dec_hidden, enc_output)

                autoencoder_loss += self.autoencoder_loss_function(targ[:, t],
                                                                   predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            loss = autoencoder_loss + self.alpha * surrogate_loss

        batch_loss = (autoencoder_loss / int(targ.shape[1]))
        gradients, variables = self.backward(loss, tape)
        self.update(gradients, variables)

        return batch_loss

    def backward(self, loss, tape):
        variables = self.enc.trainable_variables + self.dec.trainable_variables + self.surrogate.trainable_variables
        gradients = tape.gradient(loss, variables)
        return gradients, variables

    def update(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def surrogate_breed(self, output, latent, tape):
        gradients = tape.gradient(output, latent)
        return gradients

    def update_latent(self, latent, gradients, eta):
        latent += eta * gradients
        return latent

    def autoencoder_loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def surrogate_loss_function(self, real, pred):
        loss_ = tf.keras.losses.mean_squared_error(real, pred)
        return tf.reduce_mean(loss_)

    def __train(self):
        dataset = Dataset()

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.enc.initialize_hidden_state()
            enc_cell = self.enc.initialize_cell_state()
            total_loss = 0

            data_generator = dataset(self.steps_per_epoch)

            for (batch, (inp, targ, targ_surrogate)) in enumerate(
                    data_generator):
                batch_loss = self.train_step(inp, targ,
                                             targ_surrogate,
                                             enc_hidden,
                                             enc_cell)
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

    def _gen_offspring(self, candidate):
        enc_hidden = self.enc.initialize_hidden_state(batch_sz=1)
        enc_cell = self.enc.initialize_cell_state(batch_sz=1)
        offspring = candidate
        eta = 0
        while True:
            eta += 1
            with tf.GradientTape() as tape:
                enc_output, enc_hidden, enc_cell = self.enc(
                    offspring, [enc_hidden, enc_cell])

                surrogate_output = self.surrogate(enc_hidden)
                gradients = self.surrogate_breed(surrogate_output, enc_hidden,
                                                 tape)
                enc_hidden = self.update_latent(enc_hidden, gradients, eta=eta)

                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([0],
                                           1)  # TODO: Change [0] to start token
                offspring = [dec_input[0, 0].numpy()]
                for t in range(1, candidate.shape[1]):
                    predictions, dec_hidden, _, _ = self.dec(
                        dec_input, dec_hidden, enc_output)
                    offspring.append(tf.argmax(predictions[0]).numpy())
                offspring = tf.expand_dims(tf.convert_to_tensor(offspring, dtype=tf.int64), axis=0)
                if not tf.math.equal(offspring, candidate).numpy().flatten().all():
                    return offspring

    def model_update(self):
        self.__train()

    def breed(self):
        # Simulate population
        dataset = Dataset()
        data_generator = dataset(1)
        for (batch, (inp, targ, targ_surrogate)) in enumerate(
                data_generator):
            # Take one sample from batch
            print(inp[:1].shape)
            new = self._gen_offspring(inp[:1])
            print(new.shape)


if __name__ == "__main__":
    neo = NeoOriginal(epochs=15)
    neo.model_update()  # second call to check epoch decay
    neo.breed()


