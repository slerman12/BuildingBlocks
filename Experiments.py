from Tasks import Gym, MNIST
from Configs import Config, mnist_small as config
from Main import run
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # task = Gym("CartPole-v0", num_runs=1, episode_len=200)
    # task_name = "CartPole_Crossover_Only"
    task = MNIST(batch_size=100)
    task_name = "MNIST_With_Mutation"

    config = Config(layer_sizes=config.LAYER_SIZES, layer_num_genes=config.LAYER_NUM_GENES,
                    population_size=config.POPULATION_SIZE, mutation_prob=0.02,
                    num_selected=config.NUM_SELECTED, num_generations=151, beta=config.BETA)

    epochs, generations, max_fitnesses, mean_fitnesses, med_fitnesses, min_fitnesses, num_intersecting, test_accs = \
        run(task, config)

    _, ax = plt.subplots()

    ax.plot(generations, max_fitnesses, label="Max Fitness")
    ax.plot(generations, mean_fitnesses, label="Mean Fitness")
    ax.plot(generations, med_fitnesses, label="Median Fitness")
    ax.plot(generations, min_fitnesses, label="Min Fitness")
    if "MNIST" in task_name:
        ax.plot(generations[0::5], test_accs, label="Test Accuracy")

    plt.legend()

    ax.set_xlabel('Generations')
    ax.set_title('{} - Mutation {} - Population {}'.format(task_name, config.MUTATION_PROB, config.POPULATION_SIZE))
    ax.set_ylabel('Accuracy (%)' if "MNIST" in task_name else "Reward (Out Of 200)")

    plt.savefig('Figures/{}_Fitnesses.png'.format(task_name))
    plt.close()

    _, ax = plt.subplots()

    layers = [[num_int[layer][0] / num_int[layer][1] for num_int in num_intersecting]
              for layer in range(len(num_intersecting[0]))]

    for i, layer in enumerate(layers):
        ax.plot(generations, layer, label="Layer {} ({} Genes)".format(i + 1, num_intersecting[0][i][1]))

    plt.legend()

    ax.set_xlabel('Generations')
    ax.set_title('{} Intersecting Genes For Fittest & Random'.format(task_name))
    ax.set_ylabel('Intersecting Genes (%)')

    plt.savefig('Figures/{}_Intersecting.png'.format(task_name))
    plt.close()

