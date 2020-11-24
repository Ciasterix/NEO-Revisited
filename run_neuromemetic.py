import random

from deap import tools

import benchmarks


def memetic_algorithm(population, toolbox, ngen, model, stats=None,
              halloffame=None, verbose=__debug__):
    """This function is a modified version of eaSimple from deap library
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param model: Neo neural model to update and breed
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

    epochs = 200

    # Begin the generational process
    for gen in range(1, ngen + 1):
        max((epochs - 1, 10))
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Training neural model
        model.update(offspring, epochs)

        # Breeding neural model
        model.breed(offspring)

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


if __name__ == "__main__":
    POP_SIZE = 1000
    NUM_GEN = 200
    IN_PARAM = 6

    random.seed(0)

    pset = benchmarks.standard_boolean_pset(IN_PARAM)
    toolbox = benchmarks.standard_toolbox(pset)
    ev = benchmarks.Maj(pset, IN_PARAM)
    toolbox.register("evaluate", ev)
    toolbox.register("evaluate", ev)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = benchmarks.standard_statistics()

    # memetic_algorithm(pop, toolbox, NUM_GEN, naural_model, stats, hof)