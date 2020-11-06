from Population import Population
from Tasks import Gym, MNIST
from Configs import Config, mnist_small as c
import numpy as np


def run(task, config):
    config(task)

    population = Population(config, task)

    epochs = []
    generations = []
    max_fitnesses = []
    mean_fitnesses = []
    med_fitnesses = []
    min_fitnesses = []
    num_intersecting = []
    test_accs = []

    for generation in range(config.NUM_GENERATIONS):
        fitness_scores = [chromosome.fitness for chromosome in population.population]
        epochs.append(task.epoch)
        generations.append(generation)
        max_fitnesses.append(population.elite_fitness)
        mean_fitnesses.append(np.mean(fitness_scores))
        med_fitnesses.append(np.median(fitness_scores))
        min_fitnesses.append(np.min(fitness_scores))
        print("\nEpochs: ", epochs[-1], "Generations: ", generations[-1])
        print("Max Fitness: ", max_fitnesses[-1])
        print("Mean Fitness: ", mean_fitnesses[-1])
        print("Median Fitness: ", med_fitnesses[-1])
        print("Min Fitness: ", min_fitnesses[-1])
        c1 = population.elite
        c2 = population.population[np.random.randint(len(population.population) - 1)]
        num_int = []
        print("Num Intersecting Genes For Fittest & Random:")
        for i, layer in enumerate(c1.layers):
            num_int.append((np.intersect1d(layer.random_seeds, c2.layers[i].random_seeds).shape[0],
                            layer.random_seeds.shape[0]))
            print("Layer {}: {}/{}".format(i + 1, num_int[-1][0], num_int[-1][1]))
        num_intersecting.append(num_int)
        if generation % 5 == 0:
            c1.generate()
            test_accs.append(task.run(c1, test=True, vectorize=False))
            print("Test Acc Of Fittest: ", test_accs[-1])
            c1.compress()
        population.evolve()

    task.terminate()

    return epochs, generations, max_fitnesses, mean_fitnesses, med_fitnesses, min_fitnesses, num_intersecting, test_accs


if __name__ == "__main__":
    # t = Gym("Pong-v0", num_runs=1, episode_len=200)
    t = MNIST(batch_size=100)

    c = Config(layer_sizes=c.LAYER_SIZES, layer_num_genes=c.LAYER_NUM_GENES,
               population_size=c.POPULATION_SIZE, mutation_prob=c.MUTATION_PROB,
               num_selected=c.NUM_SELECTED, num_generations=c.NUM_GENERATIONS, beta=c.BETA)
    run(t, c)
