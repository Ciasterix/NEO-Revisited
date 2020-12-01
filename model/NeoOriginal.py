import time

import deap
import tensorflow as tf

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Population import Population
from model.Surrogate import Surrogate


class NeoOriginal:

    def __init__(  # TODO move parameters to config file
            self,
            pset,
            batch_size=64,
            max_size=100,
            vocab_inp_size=32,
            vocab_tar_size=32,
            embedding_dim=64,
            units=128,
            hidden_size=128,
            alpha=0.1,
            epochs=200,
            epoch_decay=1,
            min_epochs=10,
            verbose=True
    ):
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_size = max_size
        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.min_epochs = min_epochs

        self.verbose = verbose

        self.enc = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        self.dec = Decoder(vocab_tar_size, embedding_dim, units, batch_size)
        self.surrogate = Surrogate(hidden_size)
        self.population = Population(pset, max_size, batch_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    # @tf.function
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
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # print(t)
                # passing enc_output to the decoder
                predictions, dec_hidden, _, _ = self.dec(
                    dec_input, dec_hidden, enc_output)

                autoencoder_loss += self.autoencoder_loss_function(targ[:, t],
                                                                   predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            loss = autoencoder_loss + self.alpha * surrogate_loss
        print("AE loss:", autoencoder_loss.numpy())
        print("Surrogate loss:", surrogate_loss.numpy())

        batch_loss = (autoencoder_loss / int(targ.shape[1]))
        gradients, variables = self.backward(loss, tape)
        self.optimize(gradients, variables)
        # print("Koniec train stepa")

        return batch_loss

    def backward(self, loss, tape):
        variables = self.enc.trainable_variables + self.dec.trainable_variables + self.surrogate.trainable_variables
        gradients = tape.gradient(loss, variables)
        return gradients, variables

    def optimize(self, gradients, variables):
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

        for epoch in range(self.epochs):
            start = time.time()

            total_loss = 0

            data_generator = self.population()
            for (batch, (inp, targ, targ_surrogate)) in enumerate(
                    data_generator):
                # print("Batch:", batch)
                # print("Shapes:", inp.shape, targ_surrogate.shape)
                enc_hidden = self.enc.initialize_hidden_state(batch_sz=len(inp))
                enc_cell = self.enc.initialize_cell_state(batch_sz=len(inp))
                batch_loss = self.train_step(inp, targ,
                                             targ_surrogate,
                                             enc_hidden,
                                             enc_cell)
                total_loss += batch_loss

                # if batch % 1 == 0 and self.verbose:
                print('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy()))

            if self.verbose:
                print('Epoch {} Loss {:.4f}'.format(
                    epoch + 1, total_loss / self.population.steps_per_epoch))
                print('Time for epoch {} sec\n'.format(time.time() - start))

        # decrease number of epoch, but don't go below self.min_epochs
        self.epochs = max(self.epochs - self.epoch_decay, self.min_epochs)

    def _gen_child(self, candidate):
        enc_hidden = self.enc.initialize_hidden_state(batch_sz=1)
        enc_cell = self.enc.initialize_cell_state(batch_sz=1)
        child = candidate
        # eta = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_cell = self.enc(
                child, [enc_hidden, enc_cell])
            for eta in range(1, 101):
                surrogate_output = self.surrogate(enc_hidden)
                gradients = self.surrogate_breed(surrogate_output, enc_hidden,
                                                 tape)
                enc_hidden = self.update_latent(enc_hidden, gradients, eta=eta)

                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([1],  # [1] - start token
                                           1)
                child = [dec_input[0, 0].numpy()]
                for t in range(1, self.max_size - 1):
                    predictions, dec_hidden, _, _ = self.dec(
                        dec_input, dec_hidden, enc_output)
                    pred_idx = tf.argmax(predictions[0]).numpy()
                    child.append(pred_idx)
                    if pred_idx == 2:
                        break
                # child =
                if child[-1] != 2:
                    child.append(2)
                child.extend([0] * (self.max_size - len(child)))
                if not tf.math.equal(tf.expand_dims(
                        tf.convert_to_tensor(child, dtype=tf.int32), axis=0),
                        candidate).numpy().flatten().all():
                    return child

    def update(self):
        self.__train()

    def breed(self):
        # Simulate population
        data_generator = self.population(batch_size=1)
        tokenized_pop = []
        for (batch, (inp, _, _)) in enumerate(data_generator):
            tokenized_pop.append(self._gen_child(inp))

        cos1 = [self.population.tokenizer.reproduce_expression(tp) for tp in
                tokenized_pop]
        offspring = [deap.creator.Individual(tp) for tp in cos1]
        return offspring


if __name__ == "__main__":
    neo = NeoOriginal(epochs=15)
    neo.update()  # second call to check epoch decay
    neo.breed()
