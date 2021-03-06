# This script is based on official example from deap library
# Source: https://github.com/DEAP/deap/blob/master/examples/gp/multiplexer.py

import random

from deap import algorithms
from deap import tools

import benchmarks


def single_test():
    POP_SIZE = 1000
    NUM_GEN = 200

    random.seed(0)

    pset = benchmarks.standard_boolean_pset(8)
    toolbox = benchmarks.standard_toolbox(pset)
    ev = benchmarks.Maj(pset, 8)
    toolbox.register("evaluate", ev)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = benchmarks.standard_statistics()

    algorithms.eaSimple(pop, toolbox, 0.8, 0.1, NUM_GEN, stats, halloffame=hof)


if __name__ == "__main__":
    tasks_list = [
        (benchmarks.Cmp, 6),
        (benchmarks.Cmp, 8),
        (benchmarks.Maj, 6),
        (benchmarks.Maj, 8),
        (benchmarks.Mux, 6),
        (benchmarks.Par, 5),
    ]
    benchmarks.test_benchmarks(
        benchmarks.eaBreakSuccessful,
        tasks_list,
        num_runs=20,
        pop_size=1000,
        num_gen=200,
        seed=0,
        verb=False
    )
