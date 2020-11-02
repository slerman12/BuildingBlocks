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


cartpole_small = Config(layer_sizes=[25, 25], layer_num_genes=[25, 25, 25], population_size=250, mutation_prob=0,
                        num_selected=25, num_generations=100000, beta=0)

cartpole_med = Config(layer_sizes=[24, 24], layer_num_genes=[25, 25, 25], population_size=1000, mutation_prob=0,
                        num_selected=50, num_generations=100000, beta=0)

cartpole_big = Config(layer_sizes=[24, 24], layer_num_genes=[100, 50, 50], population_size=1000, mutation_prob=0,
                      num_selected=50, num_generations=100000, beta=0)

pong = Config(layer_sizes=[64 ** 2, 640, 64], layer_num_genes=[500, 250, 250, 250], population_size=2000,
              mutation_prob=0, num_selected=80, num_generations=100000, beta=0)

mnist_small = Config(layer_sizes=[784, 128, 64], layer_num_genes=[75, 75, 50, 25], population_size=250,
                     mutation_prob=0.02, num_selected=25, num_generations=100000, beta=0)

mnist_big = Config(layer_sizes=[784, 128, 64], layer_num_genes=[500, 250, 250, 250], population_size=2000,
                   mutation_prob=0, num_selected=80, num_generations=100000, beta=0)
