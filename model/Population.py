from math import ceil

import numpy as np
import tensorflow as tf

from benchmarks import MajSemantic
from utils import TreeTokenizer


class Population:
    def __init__(self, pset, max_size, batch_size):
        self.batch_size = batch_size
        self.tokenizer = TreeTokenizer(pset, max_size)
        self.task_semantic = MajSemantic(pset)

    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            # steps_per_epoch = self.steps_per_epoch
        i = 0
        self.steps_per_epoch = ceil(len(self.fitness) / batch_size)
        # print(batch_size)
        while i < self.steps_per_epoch:
            i += 1
            batch_size = 4
            idx = (i - 1) * batch_size
            n_idx = i * batch_size
            input_batch = tf.constant(self.samples[idx:n_idx])
            target_batch = input_batch
            target_surrogate_batch = tf.constant(self.fitness[idx:n_idx], dtype=tf.float32)
            yield input_batch, target_batch, target_surrogate_batch

    def update(self, offspring, gen):
        self.samples = [self.tokenizer.tokenize_tree(str(p)) for p in
                         offspring]
        len = (np.array(self.samples) == 2).argmax(1) + 1
        print("Min, Mean, Max:", np.min(len), np.mean(len), np.max(len))
        # print("Update shape", np.array(self.samples).shape)
        # self.fitness = [p.fitness.values for p in offspring]
        # self.save_pop(self.samples, self.fitness, gen)
        self.fitness = [self.task_semantic(o) for o in offspring]
        # print(self.fitness)
        # self.

    def save_pop(self, samples, fitness, gen):
        with open("population/samples_{}".format(gen), "wb") as f:
            np.save(f, np.array(samples))
        with open("population/fitness_{}".format(gen), "wb") as f:
            np.save(f, np.array(fitness))

if __name__ == "__main__":
    samples = np.random.randn(1000, 64)
    fitness = np.random.randn(1000, 1)
    p = Population(samples, fitness, batch_size=213)
    epochs = 5
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        p_gen = p()
        for batch, (inp, out, fit) in enumerate(p_gen):
            print(batch, inp.shape, out.shape, fit.shape)
