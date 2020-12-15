import time

import deap
import numpy as np
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
            dec_input = tf.expand_dims([1] * self.batch_size,
                                       # [1] - starting token
                                       1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, self.max_size):
                # print(t)
                # passing enc_output to the decoder
                predictions, dec_hidden, _, _ = self.dec(
                    dec_input, dec_hidden, enc_output)

                autoencoder_loss += self.autoencoder_loss_function(targ[:, t],
                                                                   predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            loss = autoencoder_loss + self.alpha * surrogate_loss
        # print("-" * 80)
        # print("AE loss:", autoencoder_loss.numpy())
        # print("Surrogate loss:", surrogate_loss.numpy())
        # print(targ.shape[1])
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

                if batch % 1 == 0 and self.verbose:
                    print(
                        'Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch,
                                                               batch_loss.numpy()))

            if self.verbose:
                epoch_loss = total_loss / self.population.steps_per_epoch
                print('Epoch {} Loss {:.6f} Time: {}'.format(
                    epoch + 1, epoch_loss, time.time() - start))

        # decrease number of epoch, but don't go below self.min_epochs
        self.epochs = max(self.epochs - self.epoch_decay, self.min_epochs)

    def _gen_childs(self, candidates, enc_hidden, enc_output):
        children = []
        for eta in range(1, 101):
            new_children, enc_hidden = self._gen_decoded(eta, enc_hidden, enc_output)
            new_children = self.cut_seq(new_children, end_token=2)
            new_ind, copy_ind = self.find_new(new_children, candidates)
            print(len(copy_ind))
            for i in new_ind:
                children.append(new_children[i])
            if len(copy_ind) < 1:
                break
            enc_hidden = tf.gather(enc_hidden, copy_ind)
            enc_output = tf.gather(enc_output, copy_ind)
            candidates = tf.gather(candidates, copy_ind)
        return children

    def _gen_decoded(self, eta, enc_hidden, enc_output):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(enc_hidden)
            surrogate_output = self.surrogate(enc_hidden)
            gradients = self.surrogate_breed(surrogate_output, enc_hidden,
                                             tape)
        enc_hidden = self.update_latent(enc_hidden, gradients, eta=eta)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([1] * len(enc_hidden),  # [1] - start token
                                   1)
        child = dec_input
        # print("dec_input", dec_input.shape)
        for t in range(1, self.max_size - 1):
            predictions, dec_hidden, _, _ = self.dec(
                dec_input, dec_hidden, enc_output)
            # pred_idx = .numpy()
            dec_input = tf.expand_dims(tf.argmax(predictions, axis=1, output_type=tf.dtypes.int32), 1)
            child = tf.concat([child, dec_input], axis=1)
        stop_tokens = tf.expand_dims([2] * len(enc_hidden), 1)
        child = tf.concat([child,
                           stop_tokens], axis=1)
        return child.numpy(), enc_hidden

    def cut_seq(self, seq, end_token=2):
        ind = (seq == end_token).argmax(1)
        res = [d[:i + 1] for d, i in zip(seq, ind)]
        return res

    def find_new(self, seq, candidates):
        new_ind = []
        copy_ind = []
        for i, (s, c) in enumerate(zip(seq, candidates)):
            if not np.array_equal(s, c):
                new_ind.append(i)
            else:
                copy_ind.append(i)
        return new_ind, copy_ind

    def _gen_latent(self, candidates):
        enc_hidden = self.enc.initialize_hidden_state(batch_sz=len(candidates))
        enc_cell = self.enc.initialize_cell_state(batch_sz=len(candidates))
        enc_output, enc_hidden, enc_cell = self.enc(candidates,
                                                    [enc_hidden, enc_cell])
        return enc_hidden, enc_output

    def update(self):
        print("Training")
        self.__train()

    def breed(self):
        print("Breed")
        # Simulate population
        data_generator = self.population(
            batch_size=len(self.population.samples))
        tokenized_pop = []
        for (batch, (inp, _, _)) in enumerate(data_generator):
            enc_hidden, enc_output = self._gen_latent(inp)
            tokenized_pop += (self._gen_childs(inp, enc_hidden, enc_output))

        cos1 = [self.population.tokenizer.reproduce_expression(tp) for tp in
                tokenized_pop]
        offspring = [deap.creator.Individual(tp) for tp in cos1]
        return offspring


if __name__ == "__main__":
    neo = NeoOriginal(epochs=15)
    neo.update()  # second call to check epoch decay
    neo.breed()
