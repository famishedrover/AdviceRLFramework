from abc import ABC, abstractmethod

import gym
import numpy as np
import torch

class AbstractAgent(ABC):
    """
    Abstract agent for other Agents.
    """

    def __init__(self, env, args, log_config):
        """Constructor for AbstractAgent"""

        self.env = env
        self.args = args
        self.log_config = log_config

        self.env_name = env.spec.id if env.spec is not None else env.name

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
        else :
            self.is_discrete = False

    @abstractmethod
    def act(self, state):
        """
        Gives an action on the state
        :param state: state of the environment
        :return: action to take
        """
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def update_model(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    @abstractmethod
    def save_params(self):
        pass

    @abstractmethod
    def write_log(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

