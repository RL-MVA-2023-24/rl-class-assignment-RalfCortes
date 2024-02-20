from gymnasium.wrappers import TimeLimit
from utils import ReplayBuffer
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from env_hiv import HIVPatient
import pickle
from joblib import dump, load
import sklearn

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def __init__(self):

        self.action_space_n = 4
        pass

    def act(self, observation, use_random=False):

        action = self.greedy_action(self.model, observation, self.action_space_n)
        return action

    def save(self, path, i=0):
        dump(self.model, path + "model_sklearn.joblib")

    def load(self):

        self.model = load("src/model_saved/model_sklearn_colab_20022024_11H25.joblib")

    def greedy_action(self, Q, s, nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)
