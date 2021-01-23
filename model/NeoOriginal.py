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

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

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

    # @tf.function
    def train_step(self, inp, targ, targ_surrogate):
        autoencoder_loss = 0
        with tf.GradientTape(persistent=True) as tape:
            latent, mean, logvar = self.enc(inp)
            logpz = self.log_normal_pdf(latent, 0., 0.)
            logqz_x = self.log_normal_pdf(latent, mean, logvar)
            vae_loss = logpz - logqz_x

            surrogate_output = self.surrogate(latent)
            surrogate_loss = self.surrogate_loss_function(targ_surrogate,
                                                          surrogate_output)

            # dec_hidden = enc_hidden
            # dec_cell = enc_cell
            # dec_hidden = self.dec.initialize_hidden_state(len(inp))
            # dec_cell = self.dec.initialize_cell_state(len(inp))
            # context = tf.zeros(shape=[len(dec_hidden), 1, dec_hidden.shape[1]])

            dec_input = tf.expand_dims([1] * len(inp), 1)
            predictions = self.dec(latent, self.max_size-1)
            autoencoder_loss += self.autoencoder_loss_function(
                targ[:, 1:], predictions
            )

            # for t in range(1, self.max_size):
            #     predictions, states = self.dec(dec_input, latent, states)
            #     autoencoder_loss += self.autoencoder_loss_function(
            #         targ[:, t], predictions)
            #     # Probabilistic teacher forcing
            #     # (feeding the target as the next input)
            #     if tf.random.uniform(
            #             shape=[], maxval=1, dtype=tf.float32) > self.prob:
            #         dec_input = tf.expand_dims(targ[:, t], 1)
            #     else:
            #         pred_token = tf.argmax(
            #             predictions, axis=1, output_type=tf.dtypes.int32)
            #         dec_input = tf.expand_dims(pred_token, 1)

            vae_loss = 0
            loss = -tf.reduce_mean(-autoencoder_loss + vae_loss) + self.alpha * surrogate_loss

        # ae_loss_per_token = tf.reduce_mean(autoencoder_loss) / int(targ.shape[1])
        batch_loss = loss
        # batch_loss = -tf.reduce_mean(autoencoder_loss + vae_loss) + self.alpha * surrogate_loss
        batch_ae_loss = tf.reduce_mean(autoencoder_loss) / int(targ.shape[1])
        batch_vae_loss = tf.reduce_mean(vae_loss)
        batch_surrogate_loss = surrogate_loss

        gradients, variables = self.backward(loss, tape)
        # gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
        self.optimize(gradients, variables)
        # print(batch_loss.shape, batch_ae_loss.shape, batch_vae_loss.shape, batch_surrogate_loss.shape)
        return batch_loss, batch_ae_loss, batch_vae_loss, batch_surrogate_loss

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

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def kl_loss(self, mean, logvar):
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - mean ** 2 - tf.exp(logvar))

        # sigma_sq_enc = tf.square(tf.exp(logvar))
        # kl_loss = -.5 * tf.reduce_mean(tf.reduce_sum(
        #     (1 + tf.math.log(1e-10 + sigma_sq_enc)) - tf.square(
        #         mean) - sigma_sq_enc, axis=1), axis=0)
        return kl_loss

    def autoencoder_loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        # loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        # loss_ *= mask
        # loss_1 *= mask
        # print("l", tf.reduce_sum(loss_))
        # print("l1", tf.reduce_sum(loss_1))
        # return tf.reduce_mean(loss_)
        return loss_

    def surrogate_loss_function(self, real, pred):
        loss_ = tf.keras.losses.mean_squared_error(real, pred)
        return tf.reduce_mean(loss_)

    def __train(self):

        for epoch in range(self.epochs):
            self.epoch = epoch
            start = time.time()

            total_loss = 0
            total_ae_loss = 0
            total_vae_loss = 0
            total_surrogate_loss = 0

            data_generator = self.population()
            for (batch, (inp, targ, targ_surrogate)) in enumerate(
                    data_generator):
                batch_loss, batch_ae_loss, batch_vae_loss, batch_surr_loss = self.train_step(
                    inp, targ, targ_surrogate)
                total_loss += batch_loss
                total_ae_loss += batch_ae_loss
                total_vae_loss += batch_vae_loss
                total_surrogate_loss += batch_surr_loss

                if False and self.verbose:
                    print(f'Epoch {epoch + 1} Batch {batch} '
                          f'Loss {batch_loss.numpy():.4f}')

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                epoch_loss = total_loss / self.population.steps_per_epoch
                ae_loss = total_ae_loss / self.population.steps_per_epoch
                vae_loss = total_vae_loss / self.population.steps_per_epoch
                surrogate_loss = \
                    total_surrogate_loss / self.population.steps_per_epoch
                epoch_time = time.time() - start
                print(f'Epoch {epoch + 1} Loss {epoch_loss:.6f} AE_loss '
                      f'{ae_loss:.6f} VAE_loss '
                      f'{vae_loss:.6f} Surrogate_loss '
                      f'{surrogate_loss:.6f} Time: {epoch_time:.3f}')

        # decrease number of epochs, but don't go below self.min_epochs
        self.epochs = max(self.epochs - self.epoch_decay, self.min_epochs)

    def _gen_children(
            self, candidates, latent, max_eta=1000):
        children = []
        eta = 0
        last_copy_ind = len(candidates)
        while eta < max_eta:
            eta += 1
            start = time.time()
            new_children = self._gen_decoded(eta, latent).numpy()
            new_children = self.cut_seq(new_children, end_token=2)
            new_ind, copy_ind = self.find_new(new_children, candidates)
            if len(copy_ind) < last_copy_ind:
                last_copy_ind = len(copy_ind)
                print("Eta {} Not-changed {} Time: {:.3f}".format(
                    eta, len(copy_ind), time.time() - start))
            for i in new_ind:
                children.append(new_children[i])
            if len(copy_ind) < 1:
                break
            latent = tf.gather(latent, copy_ind)
            # enc_cell = tf.gather(enc_cell, copy_ind)
            # enc_states = [enc_hidden, enc_cell]
            candidates = tf.gather(candidates, copy_ind)
        if eta == max_eta:
            print("Maximal value of eta reached - breed stopped")
        for i in copy_ind:
            children.append(new_children[i])
        return children

    def _gen_decoded(self, eta, latent):
        # enc_hidden, enc_cell = enc_states
        with tf.GradientTape(
                persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(latent)
            surrogate_output = self.surrogate(latent)
        gradients = self.surrogate_breed(surrogate_output, latent,
                                         tape)
        latent = self.update_latent(latent, gradients, eta=eta)
        # dec_cell = enc_cell
        # dec_cell = self.dec.initialize_cell_state(len(dec_hidden))
        # context = tf.zeros(shape=[len(dec_hidden), 1, dec_hidden.shape[1]])

        dec_input = tf.expand_dims([1] * len(latent), 1)

        # child = dec_input
        # states = [dec_hidden, dec_cell]
        predictions = self.dec(latent, self.max_size - 2)
        # predictions shape: (batch, max_size-2, layer_dim)
        predicted_tokens = tf.argmax(predictions, axis=2, output_type=tf.dtypes.int32)
        # for _ in range(1, self.max_size - 1):
        #     predictions, states = self.dec(latent, self.max_size-1)
        #     dec_input = tf.expand_dims(
        #         tf.argmax(predictions, axis=1, output_type=tf.dtypes.int32), 1)
        #     child = tf.concat([child, dec_input], axis=1)
        stop_tokens = tf.expand_dims([2] * len(latent), 1)
        child = tf.concat([dec_input,
                           predicted_tokens,
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

    def _gen_latents(self, candidates):
        latents = self.enc(candidates)
        return latents

    def update(self):
        print("Training")
        self.enc.train()
        self.dec.train()
        self.__train()
        self.save_models()
        self.train_steps += 1

    def breed(self):
        print("Breed")
        self.enc.eval()
        self.dec.eval()
        data_generator = self.population(
            batch_size=len(self.population.samples))

        tokenized_pop = []
        for (batch, (inp, _, _)) in enumerate(data_generator):
            latent = self._gen_latents(inp)

            tokenized_pop += (
                self._gen_children(inp, latent))

        pop_expressions = [
            self.population.tokenizer.reproduce_expression(tp)
            for tp in tokenized_pop
        ]
        offspring = [deap.creator.Individual(pe) for pe in pop_expressions]
        return offspring
