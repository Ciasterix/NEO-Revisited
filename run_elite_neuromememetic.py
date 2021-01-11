import random

from deap import tools

import benchmarks
from model.NeoOriginal import NeoOriginal


def save_population(offspring, path):
    with open(path, 'w') as f:
        for o in offspring:
            f.write(str(o)+'\n')


def merge_populations_elite(pop1, pop2, pop_size):
    unique = []
    for ind in pop + pop2:
        if ind not in unique:
            unique.append(ind)

    offspring = tools.selBest(unique, pop_size)

    num_missing = pop_size - len(offspring)
    if num_missing > 0:
        offspring += tools.selTournament(pop1, num_missing, tournsize=7)
    return offspring


def count_unique_individuals(pop):
    unique = []
    for ind in pop:
        if ind not in unique:
            unique.append(ind)
    return len(unique)


def memetic_algorithm(population, toolbox, ngen, model, stats=None,
                      halloffame=None, verbose=__debug__):
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

    save_population(population, f"offsprings/0_pop_start.txt")

    offspring = population[:]

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        # offspring = toolbox.select(population, len(population))
        print("Unique individuals:", count_unique_individuals(offspring))

        model.population.update(offspring)

        # Training neural model
        model.update()

        # Breeding neural model
        offspring = model.breed()

        # store offspring
        save_population(offspring, f"offsprings/{gen}.txt")

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = merge_populations_elite(
            population, offspring, len(population))

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if halloffame[0].fitness.getValues()[0] == 1:
            break

    return population, logbook


if __name__ == "__main__":
    POP_SIZE = 100
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
        batch_size=250,
        max_size=64,
        vocab_inp_size=15,
        vocab_tar_size=15,
        embedding_dim=64,
        units=128,
        hidden_size=128,
        alpha=0.8,
        epochs=200,
        epoch_decay=1,
        min_epochs=10,
        verbose=True
    )
    memetic_algorithm(pop, toolbox, NUM_GEN, neural_model, stats, hof)
