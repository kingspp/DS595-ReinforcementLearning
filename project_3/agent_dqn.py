#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from itertools import count
from agent import Agent
from dqn_model import DQN
import time
import torchvision.transforms as T
from torch.autograd import Variable

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class StatePrep(object):
    """ Preproces the state. """

    def __init__(self, prepfn, size):
        self.prepfn = prepfn
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Lambda(lambda x: x.convert('L')),
            T.Scale(size, interpolation=3),
            T.ToTensor()])

    def run(self, s):
        return self.transform(self.prepfn(s)).unsqueeze(0)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.args = args
        self.steps_done = 0
        self.env = env
        self.eps_threshold = None
        self.nA = env.action_space.n
        self.A = np.arange(self.nA)
        self.R = np.zeros(args.window, np.float32)
        self.M = np.zeros(args.window, np.float32)
        self.L = np.zeros(args.window, np.float32)
        self.cur_eps = None
        self.t = 0
        self.ep_len = 0

        self.prep = StatePrep(lambda s: s[50:, :, :], 84)

        self.dummy_input = torch.zeros((1, self.nA))
        self.dummy_batch = torch.zeros((self.args.batch_size, self.nA))

        self.policy_net = DQN(env).to(self.args.device)
        self.target_net = DQN(env).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025, eps=0.001, alpha=0.95)

        # Compute Huber loss
        self.loss = torch.nn.MSELoss()  # F.smooth_l1_loss #

        self.policy_net.train()
        self.target_net.eval()

        self.P = np.zeros(env.action_space.n, np.float32)

        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            self.policy_net.load_weights(self.args.model_load_path)
            ###########################
            # YOUR IMPLEMENTATION HERE #

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    # def make_action(self, observation, cur_eps, test=True):
    #     """
    #     Return predicted action of your agent
    #     Input:
    #         observation: np.array
    #             stack 4 last preprocessed frames, shape: (84, 84, 4)
    #     Return:
    #         action: int
    #             the predicted action from trained model
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
    #
    #     sample = random.random()
    #     self.eps_threshold = self.args.eps_min + (self.args.eps - self.args.eps_min) * \
    #                          math.exp(-1. * self.steps_done / self.args.eps_decay_window)
    #     self.steps_done += 1
    #     if sample > self.eps_threshold:
    #         with torch.no_grad():
    #             # t.max(1) will return largest column value of each row.
    #             # second column on max result is index of where max element was
    #             # found, so we pick action with the larger expected reward.
    #             pred = self.policy_net(observation).max(1)
    #             action = pred[1].view(1, 1)
    #             q = pred[0]
    #     else:
    #         action = torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.args.device,
    #                               dtype=torch.long)
    #         q = torch.tensor([0])
    #     # self.P.fill(cur_eps / self.nA)
    #     # q, argq = self.policy_net(Variable(observation, volatile=True)).data.cpu().max(1)
    #     # self.P[argq[0].item()] += 1 - cur_eps
    #     # action = torch.tensor([np.random.choice(self.A, p=self.P)])
    #     ###########################
    #     return action, q

    def make_action(self, observation, test=True, *args):
        """
        ***Add random action to avoid the testing model stucks under certain situation***
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        # print(self.policy_net(observation))
        # print('obs:' , np.argmax(self.policy_net(observation).detach().numpy()))
        # print(np.expand_dims(observation, axis=0).shape, self.dummy_input.shape)
        # exit()

        if not self.args.test_dqn:
            if self.eps >= random.random() or self.t < self.args.mem_init_size:
                action = torch.tensor(random.randrange(self.nA))
            else:
                action = self.policy_net(observation).max(1)[1].view(1, 1)#np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0), self.dummy_input])[0])
            # Anneal epsilon linearly over time
            if self.eps > self.args.eps_min and self.t >= self.args.mem_init_size:
                # print('Changing eps')
                self.eps -= 0.025
        else:
            if 0.005 >= random.random():
                action = torch.tensor(random.randrange(self.nA))
            else:
                action = self.policy_net(observation).max(1)[1].view(1, 1)#np.argmax(self.q_network.predict([np.expand_dims(observation, axis=0), self.dummy_input])[0])

        return action, torch.tensor([0])

    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) < self.args.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.args.capacity
        ###########################

    def replay_buffer(self, batch_size):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # batch = random.sample(self.memory, batch_size)
        ###########################
        return random.sample(self.memory, batch_size)
        # return map(lambda x: Variable(torch.cat(x, 0)), zip(*batch))

    # def optimize_model(self):
    #     # if len(self.memory) < self.args.batch_size:
    #     #     return
    #     # transitions = self.replay_buffer(self.args.batch_size)
    #     # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    #     # # detailed explanation). This converts batch-array of Transitions
    #     # # to Transition of batch-arrays.
    #     # batch = self.transition(*zip(*transitions))
    #     # # Compute a mask of non-final states and concatenate the batch elements
    #     # # (a final state would've been the one after which simulation ended)
    #     # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #     #                                         batch.next_state)), device=self.args.device, dtype=torch.bool)
    #     #
    #     # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #     # state_batch = torch.cat(batch.state)
    #     # action_batch = torch.cat(batch.action)
    #     # reward_batch = torch.cat(batch.reward)
    #
    #     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    #     # columns of actions taken. These are the actions which would've been taken
    #     # for each batch state according to policy_net
    #     # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    #
    #     # Compute V(s_{t+1}) for all next states.
    #     # Expected values of actions for non_final_next_states are computed based
    #     # on the "older" target_net; selecting their best reward with max(1)[0].
    #     # This is merged based on the mask, such that we'll have either the expected
    #     # state value or 0 in case the state was final.
    #     # next_state_values = torch.zeros(self.args.batch_size, device=self.args.device)
    #     # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    #     # Compute the expected Q values
    #     # expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch
    #     bs, ba, bns, br, bdone = self.replay_buffer(self.args.batch_size)
    #     bq = self.policy_net(bs).gather(1, ba.flatten().unsqueeze(1)).squeeze(1)
    #     bnq = self.target_net(bns).detach().max(1)[0].squeeze(0) * self.args.gamma * (1 - bdone.float())
    #
    #     # Compute Huber loss
    #     loss = self.loss(bq, br+bnq)
    #     # Optimize the model
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     # for param in self.policy_net.parameters():
    #     #     param.grad.data.clamp_(-1, 1)
    #     self.optimizer.step()
    #     return loss

    def optimize_model(self):
        transitions = self.replay_buffer(self.args.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.args.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        action_batch = torch.reshape(action_batch, [32, 1])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.args.batch_size, device=self.args.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch
        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def reset(self, state):
        return torch.reshape(torch.tensor(state, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1, 2)

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        self.eps = max(self.args.eps, self.args.eps_min)
        self.eps_delta = (self.eps - self.args.eps_min) / self.args.eps_decay_window

        print("Initializing buffer . . .")
        state = self.reset(self.env.reset())
        for _ in range(self.args.mem_init_size):
            action, _ = self.make_action(state, self.eps)
            next_state, reward, done, _ = self.env.step(action.item())
            next_state = self.reset(next_state)
            self.push(state, torch.tensor([int(action)]), next_state, torch.tensor([reward]), torch.tensor([done]))
            state = self.reset(self.env.reset()) if done else next_state

        self.t=0
        self.eps = self.args.eps
        print("Buffer Initialized Successfully.")
        for i_episode in range(self.args.max_episodes):
            # Initialize the environment and state
            start_time = time.time()
            state = self.reset(self.env.reset())
            self.R[i_episode % self.args.window] = 0
            self.L[i_episode % self.args.window] = 0
            self.M[i_episode % self.args.window] = -1e9
            self.ep_len = 0
            done = False
            while not done:
                if self.t % self.args.target_update == 0:
                    print("Updating target network . . .")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # Select and perform an action
                self.cur_eps = max(self.args.eps_min, self.eps - self.eps_delta * self.t)
                action, q = self.make_action(self.reset(state), self.cur_eps)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = self.reset(next_state)
                reward = torch.tensor([reward], device=self.args.device)
                self.R[i_episode % self.args.window] += reward
                self.M[i_episode % self.args.window] = max(self.M[i_episode % self.args.window], q[0].item())

                self.t += 1
                self.ep_len += 1

                # Store the transition in memory
                self.push(state, torch.tensor([int(action)]), next_state, torch.tensor([reward]),
                          torch.tensor([done]))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.ep_len % self.args.learn_freq == 0:
                    loss = self.optimize_model()
                    self.L[i_episode % self.args.window] += loss.detach().numpy()

                # Update the target network, copying all weights and biases in DQN

            print(
                f"Episode: {i_episode} ({self.t}) time: {time.time() - start_time:.2f} len: {self.ep_len} mem: {len(self.memory)}"
                f" EPS: {self.eps:.5f} R: {self.R[i_episode % self.args.window]:}, Avg_R: {np.mean(self.R):.3f}"
                f" Q: {self.M[i_episode % self.args.window]:.2f} Avg_Q:{np.mean(self.M):.2f}"
                f" Loss: {self.L[i_episode % self.args.window]:.2f}, Avg_Loss: {np.mean(self.L):.4f}")

        ###########################
