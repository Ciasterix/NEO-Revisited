import random

from deap import tools

import benchmarks
from model.NeoOriginal import NeoOriginal


def save_population(offspring, path):
    print("saving offspring to", path)
    with open(path, 'w') as f:
        for o in offspring:
            f.write(str(o) + '\n')

def save_log(log, path):
    with open(path, 'a') as f:
        f.write(str(log) + '\n')


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
        log_stream = logbook.stream
        print(log_stream)
        save_log(log_stream, "logs/vae_tmp.log")

    # save_population(population, f"offsprings/0_pop_start.txt")

    # model_name = "2021-01-28_15:05:51.292065"
    # model.load_models(model_name, 0)

    # save_population(population, f"offsprings/0_pop_start.txt")
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # max((epochs - 1, 10))
        # Select the next generation individuals
        # offspring = population
        offspring = toolbox.select(population, len(population))
        # save_population(offspring, f"offsprings/sel_{gen}.txt")
        model.population.update(offspring, gen)

        # Training neural model
        model.update()

        # Breeding neural model
        offspring = model.breed()

        # store offspring
        save_population(offspring, f"offsprings/breed_{gen}.txt")

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # try:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # except:
        #     print(ind)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            log_stream = logbook.stream
            print(log_stream)
            save_log(log_stream, "logs/vae_tmp.log")

        if halloffame[0].fitness.getValues()[0] == 1:
            break

    return population, logbook


if __name__ == "__main__":
    POP_SIZE = 1000
    NUM_GEN = 200
    IN_PARAM = 6

    random.seed(0)

    benchmarks.standard_creator()

    pset = benchmarks.standard_boolean_pset(IN_PARAM)
    toolbox = benchmarks.standard_toolbox(pset)
    ev = benchmarks.Maj(pset, IN_PARAM)
    toolbox.register("evaluate", ev)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = benchmarks.standard_statistics()
    neural_model = NeoOriginal(
        pset,
        batch_size=256,
        max_size=40,
        vocab_inp_size=15,
        vocab_tar_size=15,
        embedding_dim=64,
        units=128,
        hidden_size=256,
        alpha=0.8,
        epochs=200,
        epoch_decay=10,
        min_epochs=10,
        verbose=True
    )
    memetic_algorithm(pop, toolbox, NUM_GEN, neural_model, stats, hof)
