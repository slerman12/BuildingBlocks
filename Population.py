import random
import numpy as np


class Layer:
    def __init__(self, config, random_seeds):
        self.config = config
        self.random_seeds = random_seeds
        self.key_size = None
        self.value_size = None
        self.keys = None
        self.values = None
        self.is_generated = False

    def generate(self, key_size, value_size, norm=True):
        self.key_size = key_size
        self.value_size = value_size
        genes = np.zeros((self.random_seeds.shape[0], self.key_size + self.value_size))
        for i, seed in enumerate(self.random_seeds):
            np.random.seed(seed)
            genes[i] = np.random.uniform(size=self.key_size + self.value_size)

        self.keys = genes[:, :self.key_size]
        if norm:
            self.keys /= np.linalg.norm(self.keys, axis=1)[:, np.newaxis]
        self.values = genes[:, self.key_size:]
        self.values /= np.linalg.norm(self.values, axis=1)[:, np.newaxis]

        self.is_generated = True

        return self

    def attend(self, query, l2_dist=False):
        assert query.shape[0] == self.keys.shape[1]
        if l2_dist:
            weights = np.sqrt(np.sum((self.keys - query) ** 2, axis=1))
        else:
            weights = query.dot(self.keys.T)

        def softmax(x, temp=1.0):
            x /= temp
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # attention_vector = np.sum(softmax(weights, 0.001)[:, np.newaxis] * self.values, axis=0)
        attention_vector = self.values[np.argmax(weights)]
        return attention_vector / np.linalg.norm(attention_vector)

    def crossover(self, chromosome_2):
        np.random.seed()
        rand = np.random.uniform(size=self.random_seeds.shape)
        # Try weighted average as a kind of safe crossover-mutation
        crossover_layer = np.rint(rand).astype(int) * self.random_seeds + np.rint(1 - rand).astype(int) * chromosome_2.random_seeds
        return Layer(self.config, crossover_layer)

    def mutate(self):
        np.random.seed()
        mutated_layer = Layer(self.config, self.random_seeds.copy())
        mutation_probs = np.random.random(size=self.random_seeds.shape)
        to_mutate = mutation_probs < self.config.MUTATION_PROB
        mutated_layer.random_seeds[to_mutate] = np.random.randint(2**32 - 1, size=np.count_nonzero(to_mutate))
        return mutated_layer

    def compress(self):
        self.key_size = None
        self.value_size = None
        self.keys = None
        self.values = None
        self.is_generated = False

    def __str__(self):
        return str(self.random_seeds)


class Chromosome:
    def __init__(self, config, layers=None):
        self.config = config

        if layers is None:
            self.layers = [Layer(config, np.random.randint(2**32 - 1, size=num_genes))
                           for num_genes in config.LAYER_NUM_GENES]
        else:
            self.layers = layers

        self.fitness = 0

        self.is_generated = False

    def generate(self):
        self.layers = [layer.generate(self.config.LAYER_SIZES[i], self.config.LAYER_SIZES[i + 1], i != 0)
                       for i, layer in enumerate(self.layers)]

        self.is_generated = True

        return self

    def forward(self, inputs):
        assert (0 <= inputs).all() and (inputs <= 1).all()
        assert self.is_generated
        o_i = self.layers[0].attend(inputs, l2_dist=True)
        for layer in self.layers[1:]:
            o_i = layer.attend(o_i)
        return o_i

    def crossover(self, chromosome_2):
        crossover_layers = [layer.crossover(chromosome_2.layers[i])
                            for i, layer in enumerate(self.layers)]
        return Chromosome(self.config, crossover_layers)

    def mutate(self):
        mutated_layers = [layer.mutate() for layer in self.layers]
        return Chromosome(self.config, mutated_layers)

    def compress(self):
        for layer in self.layers:
            layer.compress()

        self.is_generated = False

    def __str__(self):
        c_str = ""
        for layer in self.layers:
            c_str += str(layer) + "\n"
        return c_str


class Population:
    def __init__(self, config, task):
        self.config = config
        self.task = task
        self.population = [Chromosome(config) for _ in range(config.POPULATION_SIZE)]
        for chromosome in self.population:
            self.fitness(chromosome)
        self.selection = self.select()
        self.elite = self.population[0]
        self.elite_fitness = self.elite.fitness

    def fitness(self, chromosome):
        chromosome.generate()
        fitness = self.task.run(chromosome)
        chromosome.compress()
        chromosome.fitness = fitness
        return fitness

    def select(self):
        self.population.sort(reverse=True, key=lambda x: x.fitness)
        self.elite = self.population[0]
        self.elite_fitness = self.elite.fitness
        return self.population[:self.config.NUM_SELECTED]

    def evolve(self):
        next_generation = []

        # Can multi-thread this for-loop (but how to create multiple instances of same task?)
        for i in range(self.config.POPULATION_SIZE):
            mother, father = random.sample(self.selection, 2)
            child = mother.crossover(father).mutate()
            self.fitness(child)
            next_generation.append(child)

        self.population = next_generation

        # TODO save elite(s)
        self.selection = self.select()
        self.task.evolve()
