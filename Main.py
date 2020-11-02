from Population import Population
from Tasks import Gym, MNIST
from Configs import Config, mnist_small as config
import numpy as np


task = Gym("CartPole-v0", num_runs=1)
task = MNIST(batch_size=100)

config = Config(layer_sizes=config.LAYER_SIZES, layer_num_genes=config.LAYER_NUM_GENES,
                population_size=config.POPULATION_SIZE, mutation_prob=config.MUTATION_PROB,
                num_selected=config.NUM_SELECTED, num_generations=config.NUM_GENERATIONS, beta=.9)

config(task)

population = Population(config, task)

for generation in range(config.NUM_GENERATIONS):
    fitness_scores = [chromosome.fitness for chromosome in population.population]
    print("\nEpochs: ", task.epoch, "Generations: ", generation)
    print("Max Fitness: ", population.elite_fitness)
    print("Mean Fitness: ", np.mean(fitness_scores))
    print("Median Fitness: ", np.median(fitness_scores))
    print("Min Fitness: ", np.min(fitness_scores))
    c1 = population.elite
    c2 = population.population[np.random.randint(len(population.population) - 1)]
    print("Num Intersecting Genes For Fittest & Random:")
    for i, layer in enumerate(c1.layers):
        print("Layer {}: {}/{}".format(i + 1, np.intersect1d(layer.random_seeds, c2.layers[i].random_seeds).shape[0],
                                       layer.random_seeds.shape[0]))
    # print("Num Unique Genes In Fittest:")
    # for i, layer in enumerate(c1.layers):
    #     print("Layer {}: {}/{}".format(i + 1, np.unique(layer.random_seeds).shape[0], layer.random_seeds.shape[0]))
    if generation % 5 == 0:
        c1.generate()
        print("Test Acc Of Fittest: ", task.run(c1, test=True, vectorize=False))
        c1.compress()
    # c2.generate()
    # print("Test Acc Of Random: ", task.run(c2, test=True, vectorize=False))
    # c2.compress()
    population.evolve()

task.terminate()
