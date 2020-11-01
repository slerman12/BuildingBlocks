import numpy as np
import gym
from gym import wrappers, logger
import cv2
import torch
from torchvision import datasets, transforms


class Gym:
    def __init__(self, env_id='CartPole-v0', outdir='/tmp/atari-agent-results', num_runs=1):
        # logger.set_level(logger.INFO)

        self.env_id = env_id
        self.env = gym.make(self.env_id)
        # self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        self.env.seed(0)

        self.num_runs = num_runs

        self.inputs_dim = np.prod(self.env.observation_space.shape)
        self.outputs_dim = self.env.action_space.n

    def observation_map(self, obs):
        if self.env_id == 'CartPole-v0':
            obs[0] /= 2.5
            obs[1] /= 2.5
            obs[2] /= 0.2
            obs[3] /= 2.5
            obs /= 2.2
            obs += 0.5
        else:
            obs = cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

        return np.clip(obs, 0, 1).flatten()

    def run(self, agent, test=False):
        total_reward = 0
        for _ in range(self.num_runs):
            obs = self.env.reset()
            while True:
                obs = self.observation_map(obs)
                forward = agent.forward(obs)
                action = np.argmax(forward)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break
        return total_reward / self.num_runs

    def evolve(self):
        pass

    def terminate(self):
        self.env.close()


class MNIST:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            # transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])
        dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('../data', train=False, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
        self.test_loader = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2))

        self.x, self.y = next(iter(self.train_loader))

        self.inputs_dim = np.prod(self.x.shape[1:])
        self.outputs_dim = 10

    def observation_map(self, obs):
        return np.clip(obs, 0, 1).flatten()

    def run(self, model, test=False):
        accuracy = 0

        for i in range(self.x.shape[0]):
            obs = self.observation_map(self.x[i].numpy())
            forward = model.forward(obs)
            if self.y[i] == np.argmax(forward):
                accuracy += 1.0

        accuracy /= self.x.shape[0]

        return accuracy

    def evolve(self):
        self.x, self.y = next(iter(self.train_loader))

    def terminate(self):
        pass
