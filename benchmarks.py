import operator
import random
import time
from itertools import product

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import gp
from deap import tools
from tqdm import trange


class Mux:
    # This class is based on the official example from the deap library
    # Source: https://github.com/DEAP/deap/blob/master/examples/gp/multiplexer.py

    def __init__(self, pset, num_in=6):
        self.pset = pset
        if num_in == 6:
            self.select_lines = 2
        elif num_in == 8:
            self.select_lines = 3
        self.total_lines = num_in
        self.total_size = 2 ** num_in
        self.data_lines = 2 ** self.select_lines
        self.inputs, self.outputs = self._create_inputs_outputs()

    def _create_inputs_outputs(self):
        # input : [A0 A1 A2 D0 D1 D2 D3 D4 D5 D6 D7] for a 8-3 mux
        inputs = [[0] * self.total_lines for i in range(2 ** self.total_lines)]
        outputs = [None] * (2 ** self.total_lines)

        for i in range(2 ** self.total_lines):
            value = i
            divisor = 2 ** self.total_lines
            # Fill the input bits
            for j in range(self.total_lines):
                divisor /= 2
                if value >= divisor:
                    inputs[i][j] = 1
                    value -= divisor
            # Determine the corresponding output
            indexOutput = self.select_lines
            for j, k in enumerate(inputs[i][:self.select_lines]):
                indexOutput += k * 2 ** j
            outputs[i] = inputs[i][indexOutput]

        return inputs, outputs

    def __call__(self, individual):
        func = gp.compile(individual, self.pset)
        return sum(func(*in_) == out for in_, out in zip(self.inputs, self.outputs)) / self.total_size,


class Par:
# This class is based on the official example from the deap library
# Source: https://github.com/DEAP/deap/blob/master/examples/gp/multiplexer.py

    def __init__(self, pset, num_in=5):
        self.pset = pset
        self.fanin_m = num_in
        self.total_size = 2 ** self.fanin_m
        self.inputs, self.outputs = self._create_inputs_outputs()

    def _create_inputs_outputs(self):
        inputs = [None] * self.total_size
        outputs = [None] * self.total_size

        for i in range(self.total_size):
            inputs[i] = [None] * self.fanin_m
            value = i
            dividor = self.total_size
            parity = 1
            for j in range(self.fanin_m ):
                dividor /= 2
                if value >= dividor:
                    inputs[i][j] = 1
                    parity = int(not parity)
                    value -= dividor
                else:
                    inputs[i][j] = 0
            outputs[i] = parity

        return inputs, outputs

    def __call__(self, individual):
        func = gp.compile(individual, self.pset)
        return sum(func(*in_) == out for in_, out in zip(self.inputs, self.outputs)) / self.total_size,


class Cmp:

    def __init__(self, pset, num_in=6):
        if num_in % 2 != 0 or num_in <= 0:
            raise ValueError("num_in has to be even and bigger than 0")
        self.pset = pset
        self.num_in = num_in
        self.total_size = 2 ** self.num_in
        self.inputs = list(product([0, 1], repeat=self.num_in))
        self._create_outputs()

    def _create_outputs(self):
        mid = self.num_in//2
        numbers = [
            (
                int(''.join([str(b) for b in binary[:mid]]), 2),
                int(''.join([str(b) for b in binary[mid:]]), 2)
            )
            for binary in self.inputs
        ]
        self.outputs = [
            int(num[0] > num[1]) for num in numbers
        ]

    def __call__(self, individual):
        func = gp.compile(individual, self.pset)
        return sum(func(*in_) == out for in_, out in zip(self.inputs, self.outputs)) / self.total_size,


class Maj:

    def __init__(self, pset, num_in=6):
        if num_in <= 0:
            raise ValueError("num_in has to be bigger than 0")
        self.pset = pset
        self.num_in = num_in
        self.total_size = 2 ** self.num_in
        self.inputs = list(product([0, 1], repeat=self.num_in))
        self._create_outputs()

    def _create_outputs(self):
        mid = self.num_in//2
        self.outputs = [bool(sum(inp) > mid) for inp in self.inputs]

    def __call__(self, individual):
        func = gp.compile(individual, self.pset)
        return sum(func(*in_) == out for in_, out in zip(self.inputs, self.outputs)) / self.total_size,


def nand(a, b):
    return not (a and b)


def standard_boolean_pset(num_in):
    pset = gp.PrimitiveSet("MAIN", num_in, "IN")
    pset.addPrimitive(operator.and_, 2)
    pset.addPrimitive(operator.or_, 2)
    pset.addPrimitive(operator.not_, 1)
    pset.addPrimitive(nand, 2)
    pset.addTerminal(1)
    pset.addTerminal(0)
    return pset


def standard_algebra_pset(num_in):
    raise NotImplementedError


def standard_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def standard_toolbox(primitives_set):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitives_set, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitives_set)

    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives_set)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))

    return toolbox


def standard_statistics():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def eaBreakSuccessful(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    """This function is a modified version of eaSimple from deap library
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if halloffame[0].fitness.getValues()[0] == 1:
            break

    return population, logbook


def test_benchmarks(algorithm, benchmarks_list, num_runs=20, pop_size=1000,
                    num_gen=200, seed=0, verb=False):
    """
    :param benchmarks_list: list of tests in a from of
           tuple (test_class, input_parameter)
    :param num_runs: how many times each test should be run
    :param pop_size: size of population
    :param num_gen: number of generations in one run
    :return:
    """

    standard_creator()

    for task_class, in_param in benchmarks_list:
        if seed is not None:
            random.seed(seed)

        task_name = task_class.__name__
        print(f"Testing benchmark {task_name} with {in_param} variables", flush=True)

        pset = standard_boolean_pset(in_param)
        toolbox = standard_toolbox(pset)
        ev = task_class(pset, in_param)
        toolbox.register("evaluate", ev)

        run_results = []
        run_times = []
        for _ in trange(num_runs, desc=f"{task_name}{in_param}"):
            start_time = time.time()

            pop = toolbox.population(n=pop_size)
            hof = tools.HallOfFame(1)
            stats = standard_statistics()
            algorithm(pop, toolbox, 0.8, 0.1, num_gen, stats,
                      halloffame=hof, verbose=verb)

            # print("Best in run:", hof[0].fitness.getValues()[0])
            run_times.append(time.time() - start_time)
            run_results.append(hof[0].fitness.getValues()[0])
        success_rate = sum([r == 1.00 for r in run_results]) / num_runs
        avg_best_run_fit = sum(run_results) / num_runs
        min_best_run_fit = min(run_results)
        avg_run_time = sum(run_times) / num_runs

        print(f"Results for {task_name} with {in_param} variables:", flush=True)
        print(f"Success rate: {success_rate}", flush=True)
        print(f"Average best run fit: {avg_best_run_fit}", flush=True)
        print(f"Min best run fit: {min_best_run_fit}", flush=True)
        print(f"Average run time: {avg_run_time}\n", flush=True)

        # print(f"\nSuccess rate of {task_name} with " +
        #       f"{in_param} variables: {success_rate}\n", flush=True)
