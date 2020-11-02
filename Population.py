import random
import numpy as np


class Layer:
    def __init__(self, config, random_seeds, running_fitness=None):
        self.config = config
        self.random_seeds = random_seeds
        self.running_fitness = running_fitness
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

    def attend(self, query, l2_dist=False, batch_dim=False):
        assert query.shape[1 if batch_dim else 0] == self.keys.shape[1]
        if l2_dist:
            if batch_dim:
                weights = np.sqrt(np.sum((self.keys[None, :, :] - query[:, None, :]) ** 2, axis=2))
            else:
                weights = np.sqrt(np.sum((self.keys - query) ** 2, axis=1))
        else:
            weights = query.dot(self.keys.T)

        def softmax(x, temp=1.0):
            x /= temp
            if batch_dim:
                e_x = np.exp(x - np.max(x, axis=1)[:, None])
                return e_x / e_x.sum(axis=1)[:, None]
            else:
                e_x = np.exp(x - np.max())
                return e_x / e_x.sum()

        if batch_dim:
            # attention_vector = np.sum(softmax(weights, 0.001)[:, :, None] * self.values[None, :, :], axis=1)
            attention_vector = self.values[np.argmax(weights, axis=1)]
            return attention_vector / np.linalg.norm(attention_vector, axis=1)[:, None]
        else:
            # attention_vector = np.sum(softmax(weights, 0.001)[:, np.newaxis] * self.values, axis=0)
            attention_vector = self.values[np.argmax(weights)]
            return attention_vector / np.linalg.norm(attention_vector)

    def crossover(self, layer_2):
        np.random.seed()
        rand = np.random.uniform(size=self.random_seeds.shape)
        # TODO Try weighted average as a kind of safe crossover-mutation
        # TODO Should the random seeds be shuffled first?
        crossover_genes = np.rint(rand).astype(int) * self.random_seeds + np.rint(1 - rand).astype(int) * layer_2.random_seeds
        if self.config.BETA != 0:
            crossover_fitness = np.rint(rand).astype(int) * self.running_fitness + np.rint(1 - rand).astype(int) * layer_2.running_fitness
        else:
            crossover_fitness = None
        return Layer(self.config, crossover_genes, crossover_fitness)

    def mutate(self):
        np.random.seed()
        mutated_layer = Layer(self.config, self.random_seeds.copy(),
                              self.running_fitness.copy() if self.config.BETA != 0 else None)
        mutation_probs = np.random.random(size=self.random_seeds.shape)
        to_mutate = mutation_probs < self.config.MUTATION_PROB
        mutated_layer.random_seeds[to_mutate] = np.random.randint(2**32 - 1, size=np.count_nonzero(to_mutate))
        if self.config.BETA != 0:
            mutated_layer.running_fitness[to_mutate] = -1
        return mutated_layer

    def compress(self):
        self.key_size = None
        self.value_size = None
        self.keys = None
        self.values = None
        self.is_generated = False

    def update_running_fitness(self, fitness):
        if self.config.BETA != 0:
            if self.running_fitness is None:
                self.running_fitness = np.full(self.random_seeds.shape, fitness)
            else:
                mutated_genes = self.running_fitness < 0
                self.running_fitness[mutated_genes] = fitness
                self.running_fitness = self.config.BETA * self.running_fitness + (1 - self.config.BETA) * fitness
        else:
            self.running_fitness = fitness

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

    def forward(self, inputs, batch_dim=False):
        assert (0 <= inputs).all() and (inputs <= 1).all()
        assert self.is_generated
        o_i = self.layers[0].attend(inputs, l2_dist=True, batch_dim=batch_dim)
        for layer in self.layers[1:]:
            o_i = layer.attend(o_i, batch_dim=batch_dim)
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

    def update_fitness(self, fitness):
        if self.config.BETA == 0:
            self.fitness = fitness
        else:
            fitnesses = 0
            for layer in self.layers:
                layer.update_running_fitness(fitness)
                fitnesses += np.sum(layer.running_fitness)
            self.fitness = fitnesses / np.sum(self.config.LAYER_NUM_GENES)

    def __str__(self):
        chromosome_str = ""
        for layer in self.layers:
            chromosome_str += str(layer) + "\n"
        return chromosome_str


class Population:
    def __init__(self, config, task):
        self.config = config
        self.task = task
        self.population = [Chromosome(config) for _ in range(config.POPULATION_SIZE)]
        for chromosome in self.population:
            self.fitness(chromosome)
        self.select()

    def fitness(self, chromosome):
        self.task.update()
        chromosome.generate()
        fitness = self.task.run(chromosome)
        chromosome.compress()
        chromosome.update_fitness(fitness)
        return fitness

    def select(self):
        self.population.sort(reverse=True, key=lambda x: x.fitness)
        self.elite = self.population[0]
        self.elite_fitness = self.elite.fitness
        self.selection = self.population[:self.config.NUM_SELECTED]

    def evolve(self):
        next_generation = []

        # Can multi-thread this for-loop
        for i in range(self.config.POPULATION_SIZE):
            mother, father = random.sample(self.selection, 2)
            child = mother.crossover(father).mutate()
            self.fitness(child)
            next_generation.append(child)

        self.population = next_generation

        # TODO Save elite(s)
        self.select()
        # self.task.update()
