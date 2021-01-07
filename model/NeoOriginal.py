import time

import deap
import numpy as np
import tensorflow as tf

from model.Decoder import Decoder
from model.Encoder import Encoder
from model.Population import Population
from model.Surrogate import Surrogate
from utils import create_expression_tree


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
        self.train_steps = 0

        self.verbose = verbose

        self.enc = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
        self.dec = Decoder(
            vocab_inp_size, vocab_tar_size, embedding_dim, units, batch_size)
        self.surrogate = Surrogate(hidden_size)
        self.population = Population(pset, max_size, batch_size)
        self.prob = 0.5

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction='none')

    def save_models(self):
        self.enc.save_weights(
            "model/weights/encoder/enc_{}".format(self.train_steps),
            save_format="tf")
        self.dec.save_weights(
            "model/weights/decoder/dec_{}".format(self.train_steps),
            save_format="tf")
        self.surrogate.save_weights(
            "model/weights/surrogate/surrogate_{}".format(self.train_steps),
            save_format="tf")

    def load_models(self, train_steps):
        self.enc.load_weights(
            "model/weights/encoder/enc_{}".format(train_steps))
        self.dec.load_weights(
            "model/weights/decoder/dec_{}".format(train_steps))
        self.surrogate.load_weights(
            "model/weights/surrogate/surrogate_{}".format(train_steps))

    @tf.function
    def train_step(self, inp, targ, targ_surrogate, enc_hidden,
                   enc_cell):
        autoencoder_loss = 0
        with tf.GradientTape(persistent=True) as tape:
            enc_output, enc_hidden, enc_cell = self.enc(
                inp, [enc_hidden, enc_cell])

            surrogate_output = self.surrogate(enc_hidden)
            surrogate_loss = self.surrogate_loss_function(targ_surrogate,
                                                          surrogate_output)

            dec_hidden = enc_hidden
            dec_cell = enc_cell
            context = tf.zeros(shape=[len(dec_hidden), 1, dec_hidden.shape[1]])

            dec_input = tf.expand_dims([1] * len(inp), 1)

            for t in range(1, self.max_size):
                initial_state = [dec_hidden, dec_cell]
                predictions, context, [dec_hidden, dec_cell], _ = self.dec(
                    dec_input, context, enc_output, initial_state)
                autoencoder_loss += self.autoencoder_loss_function(
                    targ[:, t], predictions)

                # Probabilistic teacher forcing
                # (feeding the target as the next input)
                if tf.random.uniform(
                        shape=[], maxval=1, dtype=tf.float32) > self.prob:
                    dec_input = tf.expand_dims(targ[:, t], 1)
                else:
                    pred_token = tf.argmax(
                        predictions, axis=1, output_type=tf.dtypes.int32)
                    dec_input = tf.expand_dims(pred_token, 1)

            loss = autoencoder_loss + self.alpha * surrogate_loss

        ae_loss_per_token = autoencoder_loss / int(targ.shape[1])
        batch_loss = ae_loss_per_token + self.alpha * surrogate_loss
        batch_ae_loss = (autoencoder_loss / int(targ.shape[1]))
        batch_surrogate_loss = surrogate_loss

        gradients, variables = self.backward(loss, tape)
        self.optimize(gradients, variables)

        return batch_loss, batch_ae_loss, batch_surrogate_loss

    def backward(self, loss, tape):
        variables = \
            self.enc.trainable_variables + self.dec.trainable_variables \
            + self.surrogate.trainable_variables
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
            self.epoch = epoch
            start = time.time()

            total_loss = 0
            total_ae_loss = 0
            total_surrogate_loss = 0

            data_generator = self.population()
            for (batch, (inp, targ, targ_surrogate)) in enumerate(
                    data_generator):
                enc_hidden = self.enc.initialize_hidden_state(batch_sz=len(inp))
                enc_cell = self.enc.initialize_cell_state(batch_sz=len(inp))
                batch_loss, batch_ae_loss, batch_surr_loss = self.train_step(
                    inp, targ, targ_surrogate, enc_hidden, enc_cell)
                total_loss += batch_loss
                total_ae_loss += batch_ae_loss
                total_surrogate_loss += batch_surr_loss

                if False and self.verbose:
                    print(f'Epoch {epoch + 1} Batch {batch} '
                          f'Loss {batch_loss.numpy():.4f}')

            if self.verbose and (epoch % 10 == 0):
                epoch_loss = total_loss / self.population.steps_per_epoch
                ae_loss = total_ae_loss / self.population.steps_per_epoch
                surrogate_loss = \
                    total_surrogate_loss / self.population.steps_per_epoch
                epoch_time = time.time() - start
                print(f'Epoch {epoch + 1} Loss {epoch_loss:.6f} AE_loss '
                      f'{ae_loss:.6f} Surrogate_loss '
                      f'{surrogate_loss:.6f} Time: {epoch_time:.3f}')

        # decrease number of epochs, but don't go below self.min_epochs
        self.epochs = max(self.epochs - self.epoch_decay, self.min_epochs)

    def _gen_children(
            self, candidates, enc_output, enc_hidden, enc_cell, max_eta=1000):
        children = []
        eta = 0
        enc_mask = enc_output._keras_mask
        while eta < max_eta:
            eta += 1
            start = time.time()
            new_children = self._gen_decoded(
                eta, enc_output, enc_hidden, enc_cell, enc_mask).numpy()
            new_children = self.cut_seq(new_children, end_token=2)
            new_ind, copy_ind = self.find_new(new_children, candidates)
            print("Eta {} Not-changed {} Time: {:.3f}".format(
                eta, len(copy_ind), time.time() - start))
            for i in new_ind:
                children.append(new_children[i])
            if len(copy_ind) < 1:
                break
            enc_output = tf.gather(enc_output, copy_ind)
            enc_mask = tf.gather(enc_mask, copy_ind)
            enc_hidden = tf.gather(enc_hidden, copy_ind)
            enc_cell = tf.gather(enc_cell, copy_ind)
            candidates = tf.gather(candidates, copy_ind)
        if eta == max_eta:
            print("Maximal value of eta reached - breed stopped")
        for i in copy_ind:
            children.append(new_children[i])
        return children

    def _gen_decoded(self, eta, enc_output, enc_hidden, enc_cell, enc_mask):
        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(enc_hidden)
            surrogate_output = self.surrogate(enc_hidden)
        gradients = self.surrogate_breed(surrogate_output, enc_hidden,
                                         tape)
        dec_hidden = self.update_latent(enc_hidden, gradients, eta=eta)
        dec_cell = enc_cell
        context = tf.zeros(shape=[len(dec_hidden), 1, dec_hidden.shape[1]])

        dec_input = tf.expand_dims([1] * len(enc_hidden), 1)

        child = dec_input
        for _ in range(1, self.max_size - 1):
            initial_state = [dec_hidden, dec_cell]
            predictions, context, [dec_hidden, dec_cell], _ = self.dec(
                dec_input, context, enc_output, initial_state, enc_mask)
            dec_input = tf.expand_dims(
                tf.argmax(predictions, axis=1, output_type=tf.dtypes.int32), 1)
            child = tf.concat([child, dec_input], axis=1)
        stop_tokens = tf.expand_dims([2] * len(enc_hidden), 1)
        child = tf.concat([child,
                           stop_tokens], axis=1)
        return child

    def cut_seq(self, seq, end_token=2):
        ind = (seq == end_token).argmax(1)
        res = []
        tree_max = []
        for d, i in zip(seq, ind):
            repaired_tree = create_expression_tree(d[:i + 1][1:-1])
            repaired_seq = [i.data for i in repaired_tree.preorder()][
                           -(self.max_size - 2):]
            tree_max.append(len(repaired_seq) == self.max_size - 2)
            repaired_seq = [1] + repaired_seq + [2]
            res.append(np.pad(repaired_seq, (0, self.max_size - i - 1)))
        return res

    def find_new(self, seq, candidates):
        new_ind = []
        copy_ind = []
        n = False
        cp = False
        for i, (s, c) in enumerate(zip(seq, candidates)):
            if not np.array_equal(s, c):
                if not n:
                    n = True
                new_ind.append(i)
            else:
                if not cp:
                    cp = True
                copy_ind.append(i)
        return new_ind, copy_ind

    def _gen_latent(self, candidates):
        enc_hidden = self.enc.initialize_hidden_state(batch_sz=len(candidates))
        enc_cell = self.enc.initialize_cell_state(batch_sz=len(candidates))
        enc_output, enc_hidden, enc_cell = self.enc(candidates,
                                                    [enc_hidden, enc_cell])
        return enc_output, enc_hidden, enc_cell

    def update(self):
        print("Training")
        self.enc.train()
        self.dec.train()
        self.__train()
        self.save_models()
        self.train_steps += 1

    def breed(self):
        print("Breed")
        self.dec.eval()
        data_generator = self.population(
            batch_size=len(self.population.samples))

        tokenized_pop = []
        for (batch, (inp, _, _)) in enumerate(data_generator):
            enc_output, enc_hidden, enc_cell = self._gen_latent(inp)

            tokenized_pop += (
                self._gen_children(inp, enc_output, enc_hidden, enc_cell))

        pop_expressions = [
            self.population.tokenizer.reproduce_expression(tp)
            for tp in tokenized_pop
        ]
        offspring = [deap.creator.Individual(pe) for pe in pop_expressions]
        return offspring
