class Config:
    def __init__(self, layer_sizes, layer_num_genes, population_size, mutation_prob, num_selected, num_generations, beta):
        self.LAYER_SIZES = layer_sizes
        self.LAYER_NUM_GENES = layer_num_genes
        self.POPULATION_SIZE = population_size
        self.MUTATION_PROB = mutation_prob
        self.NUM_SELECTED = num_selected
        self.NUM_GENERATIONS = num_generations
        self.BETA = beta
        self.task = None

    def set_task(self, task):
        self.task = task
        self.task_func()
        return self

    def task_func(self):
        self.LAYER_SIZES.insert(0, self.task.inputs_dim)
        self.LAYER_SIZES.append(self.task.outputs_dim)

    def __call__(self, task):
        return self.set_task(task)

    def __str__(self):
        return "Layer Sizes: {}\nLayer Num Genes: {}\nPopulation Size: {}\nMutation Prob: {}\nNum Selected: {}\n" \
               "Num Generations: {}\nBeta: {}".format(self.LAYER_SIZES, self.LAYER_NUM_GENES, self.POPULATION_SIZE,
                                                      self.MUTATION_PROB, self.NUM_SELECTED, self.NUM_GENERATIONS,
                                                      self.BETA)


cartpole_small = Config(layer_sizes=[25, 25], layer_num_genes=[25, 25, 25], population_size=250, mutation_prob=0,
                        num_selected=25, num_generations=100000, beta=0)

cartpole_med = Config(layer_sizes=[24, 24], layer_num_genes=[25, 25, 25], population_size=1000, mutation_prob=0,
                        num_selected=50, num_generations=100000, beta=0)

cartpole_big = Config(layer_sizes=[24, 24], layer_num_genes=[100, 50, 50], population_size=1000, mutation_prob=0,
                      num_selected=50, num_generations=100000, beta=0)

pong = Config(layer_sizes=[784, 64], layer_num_genes=[50, 25, 25], population_size=250,
              mutation_prob=0.01, num_selected=25, num_generations=100000, beta=0)

mnist_small = Config(layer_sizes=[784, 128, 64], layer_num_genes=[75, 75, 50, 25], population_size=250,
                     mutation_prob=0.02, num_selected=25, num_generations=100000, beta=0)

mnist_big = Config(layer_sizes=[256, 128, 64], layer_num_genes=[500, 75, 50, 25], population_size=350,
                   mutation_prob=0.02, num_selected=35, num_generations=100000, beta=0)

mnist_single_layer = Config(layer_sizes=[784], layer_num_genes=[500, 75], population_size=350,
                     mutation_prob=0.02, num_selected=35, num_generations=100000, beta=0)