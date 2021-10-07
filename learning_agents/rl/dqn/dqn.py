"""
TODO
1. Complete PER / Naive DQN loss update.
2. https://github.com/GuanSuns/EXPAND/blob/b5c87503681375abbe808daa3fd34811c1cbb185/learning_agents/rl/dqn/dqn.py#L89

"""


import time

import numpy as np
import torch
from collections import deque

from learning_agents.abstractAgent import AbstractAgent
from learning_agents.common.replay_buffers import PrioritizedReplayBuffer, ReplayBuffer
from learning_agents.common.replay_buffer.PrioritizedReplayBuffer import NaivePrioritizedBuffer

from learning_agents.architectures.nn_models import Conv2d_MLP_Model


class DQN(AbstractAgent):
    def __init__(self, env, args, log_config, hyper_params, network_config, optimizer_config, global_config,
                 logger=None):
        super(DQN, self).__init__(env, args, log_config)

        self.hyper_params = hyper_params
        self.network_config = network_config
        self.optimizer_config = optimizer_config
        self.global_config = global_config
        self.logger = logger

        self.episode_step = 0
        self.total_step = 0
        self.current_episode = 0

        self.state_dim = self.env.observation_space.shape[0]
        self.state_channels = self.hyper_params.frame_stack

        # DQN only works with discrete action spaces.
        self.is_discrete = True
        self.action_dim = self.env.action_space.n

        self.memory = None

        self.device = self.global_config.torch_device

        self._initialize()
        self._init_network()

    def _initialize(self):
        """
        Initialize Replay Buffers, and misc things
        :return:
        """

        if self.hyper_params.prioritized_relpay:
            self.memory = PrioritizedReplayBuffer(capacity=self.hyper_params.buffer_size,
                                                  alpha=self.hyper_params.prioritized_relpay_alpha)

        elif self.hyper_params.naive_prioritized_replay:
            self.memory = NaivePrioritizedBuffer(capacity=self.hyper_params.buffer_size,
                                                 alpha=self.hyper_params.prioritized_relpay_alpha)

        else:
            self.memory = ReplayBuffer(capacity=self.hyper_params.buffer_size)

    def _init_network(self):
        """
        Initialize network and optimizer
        :return:
        """

        self.dqn_network = Conv2d_MLP_Model(input_channels=self.state_channels,
                                            fc_input_size=self.network_config.fc_input_size,
                                            fc_output_size=self.network_config.fc_output_size,
                                            nonlinearity=self.network_config.nonlininearity,
                                            channels=self.network_config.channels,
                                            kernel_sizes=self.network_config.kernel_sizes,
                                            strides=self.network_config.strides,
                                            paddings=self.network_config.paddings,
                                            fc_hidden_sizes=self.network_config.fc_hidden_sizes,
                                            fc_hidden_activation=self.network_config.fc_hidden_activation
                                            ).to(self.device)

        self.dqn_target = Conv2d_MLP_Model(input_channels=self.state_channels,
                                           fc_input_size=self.network_config.fc_input_size,
                                           fc_output_size=self.network_config.fc_output_size,
                                           nonlinearity=self.network_config.nonlininearity,
                                           channels=self.network_config.channels,
                                           kernel_sizes=self.network_config.kernel_sizes,
                                           strides=self.network_config.strides,
                                           paddings=self.network_config.paddings,
                                           fc_hidden_sizes=self.network_config.fc_hidden_sizes,
                                           fc_hidden_activation=self.network_config.fc_hidden_activation
                                           ).to(self.device)

        self.dqn_target.load_state_dict(self.dqn_network.state_dict())
        for param in self.dqn_target.parameters():
            param.requires_grad = False

        # Optimizer :
        self.dqn_optimizer = torch.optim.Adam(self.dqn_network.parameters(), lr=self.optimizer_config.lr_dqn,
                                              weight_decay=self.optimizer_config.weight_decay,
                                              eps=self.optimizer_config.adam_eps)

        # Start network from a file

        self._init_network_from_file()


    def _init_network_from_file(self):
        if self.args.load_from is not None :
            self.load_params(self.args.load_from)


    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


    def load_params(self, path):
        AbstractAgent.load_params(self, path)

        params = torch.load(path, map_location=self.device)
        self.dqn_network.load_state_dict(params["dqn_network_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optimizer.load_state_dict(params["dqn_optimizer_state_dict"])

        print ("[INFO] Loaded Dqn Network, Target Network and Optimizer from ", path)



    def save_params(self, n_step):
        """
        Save model and optimizer parameters
        :return:
        """

        params = {
            "dqn_network_state_dict" : self.dqn_network.state_dict(),
            "dqn_target_state_dict" : self.dqn_target.state_dict(),
            "dqn_optimizer_state_dict" : self.dqn_optimizer.state_dict(),
        }
        AbstractAgent.save_params(self, params, n_step)

        # if self.logger is not None :
        #     self.logger.save_models(params, prefix="model", postfix=str(n_step), is_snapshot=True)


    def pretrain(self):
        """
        No pretrainnig in DQN
        :return:
        """

        pass


    def train(self):
        """Train the agent."""
        # logger
        if self.logger is not None:
            self.logger.watch_wandb([self.dqn, self.dqn_target])

        # pre-training if needed
        self.pretrain()

        avg_scores_window = deque(maxlen=self.args.avg_score_window)
        eval_scores_window = deque(maxlen=self.args.eval_score_window)

        for i_episode in range(1, self.args.iteration_num + 1):
            self.i_episode = i_episode
            self.testing = (self.i_episode % self.args.eval_period == 0)

            state = np.squeeze(self.env.reset(), axis=0)
            self.episode_step = 0
            losses = list()
            done = False
            score = 0

            states_queue = deque(maxlen=self.hyper_params.frame_stack)
            states_queue.extend([state for _ in range(self.hyper_params.frame_stack)])

            t_begin = time.time()

            while not done:
                if self.args.render \
                        and self.i_episode >= self.args.render_after \
                        and self.i_episode % self.args.render_freq == 0:
                    self.env.render()

                stacked_states = np.copy(np.stack(list(states_queue), axis=0))
                action = self.select_action(stacked_states)
                next_state, reward, done, info = self.step(action)
                # add next state into states queue
                next_state = np.squeeze(next_state, axis=0)
                states_queue.append(next_state)
                # save the new transition
                transition = (stacked_states, action, reward, np.copy(np.stack(list(states_queue), axis=0)), done, info)
                self._add_transition_to_memory(transition)

                self.total_step += 1
                self.episode_step += 1
                if self.total_step % self.args.save_period == 0:
                    self.save_params(self.total_step)

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            loss = self.update_model()
                            losses.append(loss)  # for logging
                score += reward

            self.do_post_episode_update()
            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step
            avg_scores_window.append(score)

            if self.testing:
                eval_scores_window.append(score)
                # noinspection PyStringFormat
                print('[EVAL INFO] episode: %d, total step %d, '
                      'evaluation score: %.3f, window avg: %.3f\n'
                      % (self.i_episode,
                         self.total_step,
                         score,
                         np.mean(eval_scores_window)))

                if self.logger is not None:
                    self.logger.log_wandb({
                        'eval score': score,
                        "eval window avg": np.mean(eval_scores_window),
                    }, step=self.total_step)

                self.testing = False

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost, np.mean(avg_scores_window))
                self.write_log(log_value)

        # termination
        self.env.close()
        self.save_params(self.total_step)


    def _add_transition_to_memory(self, transition):
        if transition :
            self.memory.push(transition)


    def write_log(self, log_value):
        pass
