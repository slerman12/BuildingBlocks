import time

import numpy as np
import gym
from gym import wrappers, logger
import cv2
import torch
from torchvision import datasets, transforms


class Gym:
    def __init__(self, env_id='CartPole-v0', outdir='/tmp/atari-agent-results', num_runs=1, episode_len=np.inf):
        # logger.set_level(logger.INFO)

        self.env_id = env_id
        self.env = gym.make(self.env_id)
        # self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        self.env.seed(0)

        self.num_runs = num_runs
        self.episode_len = episode_len

        self.obs = self.env.reset()
        self.obs = self.observation_map(self.obs)

        self.inputs_dim = np.prod(self.obs.shape)
        self.outputs_dim = self.env.action_space.n

        self.epoch = 0

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

    def run(self, agent, test=False, vectorize=True):
        total_reward = 0
        obs = self.obs
        for _ in range(self.num_runs):
            steps = 0
            while True and steps < self.episode_len:
                forward = agent.forward(obs)
                action = np.argmax(forward)
                obs, reward, done, _ = self.env.step(action)
                obs = self.observation_map(obs)
                total_reward += reward
                steps += 1
                if done:
                    obs = self.env.reset()
                    obs = self.observation_map(obs)
                    self.epoch += 1
                    break
        self.obs = obs
        return total_reward / self.num_runs

    def update(self):
        pass

    def terminate(self):
        self.env.close()


class MNIST:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        self.epoch = 0

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST('../data', train=False, download=True, transform=transform)

        self.train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
        self.train_iterator = iter(self.train_loader)
        self.test_loader = torch.utils.data.DataLoader(dataset2, batch_size=len(dataset2))

        self.update()
        self.x_test, self.y_test = next(iter(self.test_loader))
        self.x_test = self.x_test.numpy()
        self.y_test = self.y_test.numpy()

        self.inputs_dim = np.prod(self.x.shape[1:])
        self.outputs_dim = 10

    def observation_map(self, obs, batch_dim=False):
        if batch_dim:
            return np.clip(obs, 0, 1).reshape(obs.shape[0], np.prod(obs.shape[1:]))
        else:
            return np.clip(obs, 0, 1).flatten()

    def run(self, model, test=False, vectorize=True):
        if test:
            # TODO why is test so much worse when batches are new/unseen each time? Are they?
            x = self.x_test
            y = self.y_test
        else:
            x = self.x
            y = self.y

        if vectorize:
            # st = time.time()
            obs = self.observation_map(x, batch_dim=True)
            forward = model.forward(obs, batch_dim=True)
            accuracy = np.sum(y == np.argmax(forward, axis=1)) / x.shape[0]
            # print(time.time() - st)
        else:
            # st = time.time()
            accuracy = 0

            for i in range(x.shape[0]):
                obs = self.observation_map(x[i])
                forward = model.forward(obs)
                if y[i] == np.argmax(forward):
                    accuracy += 1.0

            accuracy /= x.shape[0]
            # print(time.time() - st)

        return accuracy

    def update(self):
        try:
            self.x, self.y = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            self.x, self.y = next(self.train_iterator)
            self.epoch += 1

        self.x = self.x.numpy()
        self.y = self.y.numpy()

    def terminate(self):
        pass
