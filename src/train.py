from gymnasium.wrappers import TimeLimit
from utils import ReplayBuffer
from copy import deepcopy
import torch
import torch.nn as nn

# from tqdm import tqdm
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

        # action = self.greedy_action(self.model, observation, self.action_space_n)

        # action = self.greedy_action_ensemble(self.list_Q_function, observation, self.action_space_n)
        action = self.greedy_action_mean(
            self.list_Q_function, observation, self.action_space_n
        )

        return action

    def save(self, path, i=0):
        dump(self.model, path + "model_sklearn.joblib")

    # def load(self):

    #     self.model = load(
    #         "src/model_saved/model_sklearn_robust_15H00_20iter_50et.joblib"
    #     )

    def load(self):
        self.list_Q_function = []
        self.list_Q_function_str = [
            "src/model_saved/model_ensemble_-1.joblib",
            f"src/model_saved/model_ensemble_{0}.joblib",
            # f"src/model_saved/model_ensemble_{3}.joblib",
            # f"src/model_saved/model_ensemble_{2}.joblib",
            # f"src/model_saved/model_ensemble_{1}.joblib",
            # f"src/model_saved/model_ensemble_{0}.joblib",
        ]

        for str_ in self.list_Q_function_str:
            self.list_Q_function.append(load(str_))

    def greedy_action(self, Q, s, nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)

    def greedy_action_ensemble(self, Q_list, s, nb_actions):

        ensemble_action = []

        for Q_ in Q_list:
            Qsa = []
            for a in range(nb_actions):
                sa = np.append(s, a).reshape(1, -1)
                Qsa.append(Q_.predict(sa))

            best_action = np.argmax(Qsa)
            ensemble_action.append(best_action)

        return max(set(ensemble_action), key=ensemble_action.count)

    def greedy_action_mean(self, Q_list, s, nb_actions):

        Qsa_store = []

        for Q_ in Q_list:
            Qsa = []
            for a in range(nb_actions):
                sa = np.append(s, a).reshape(1, -1)
                Qsa.append(Q_.predict(sa))

            # normalize Qsa
            Qsa = np.array(Qsa)
            Qsa_norm = (Qsa - np.mean(Qsa)) / np.std(Qsa)
            Qsa_store.append(Qsa_norm)

        # mean of Qsa
        Qsa_mean = np.mean(Qsa_store, axis=0)
        best_action = np.argmax(Qsa_mean)

        return best_action
