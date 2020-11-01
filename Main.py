from Population import Population
from Tasks import Gym, MNIST
import Configs
import numpy as np


task = Gym("CartPole-v0", num_runs=1)
config = Configs.cartpole_small(task)

task = MNIST(batch_size=32)
config = Configs.mnist_small(task)

population = Population(config, task)

for generation in range(config.NUM_GENERATIONS):
    fitness_scores = [chromosome.fitness for chromosome in population.population]
    print("\nGeneration: ", generation)
    print("Max Fitness: ", population.elite_fitness)
    print("Mean Fitness: ", np.mean(fitness_scores))
    print("Median Fitness: ", np.median(fitness_scores))
    c1 = population.elite
    c2 = population.population[np.random.randint(len(population.population) - 1)]
    print("Num Intersecting Genes For Fittest & Random:")
    for i, layer in enumerate(c1.layers):
        print("Layer {}: {}/{}".format(i + 1, np.intersect1d(layer.random_seeds, c2.layers[i].random_seeds).shape[0],
                                       layer.random_seeds.shape[0]))
    population.elite.generate()
    print("Test Acc Of Fittest: ", task.run(population.elite, test=True))
    population.elite.compress()
    population.evolve()

task.terminate()
