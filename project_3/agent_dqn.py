#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from itertools import count
import gc
from agent import Agent
from dqn_model import DQN
import time
from torch.autograd import Variable
import json
import uuid

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class JsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            try:
                return obj.default()
            except Exception:
                return f'Object not serializable - {obj}'


class MetaData(object):
    def __init__(self, fp, args):
        self.transition = namedtuple('Data',
                                     (
                                         "episode", "step", "time", "time_elapsed", "ep_len", "buffer_len", "epsilon",
                                         "reward",
                                         "avg_reward", "max_q", "max_avg_q",
                                         "loss", "avg_loss", "mode"))
        self.fp = fp
        self.data = None
        self.args = args

    def update(self, *args):
        self.data = self.transition(*args)
        if self.data.episode % self.args.disp_freq == 0:
            print(
                f"E: {self.data.episode} |  Step: {self.data.step} | T: {self.data.time:.2f} | ET: {self.data.time_elapsed:.2f}"
                f" | Len: {self.data.ep_len} | EPS: {self.data.epsilon:.5f} | R: {self.data.reward} | AR: {self.data.avg_reward:.3f}"
                f" | MAQ:{self.data.max_avg_q:.2f} | L: {self.data.loss:.2f} | AL: {self.data.avg_loss:.4f} | Mode: {self.data.mode}")
        self.fp.write(self.data._asdict().values().__str__().replace('odict_values([', '').replace('])', '' + '\n'))

    def load(self, f):
        self.data = self.transition(*json.load(f).values())

    def dump(self, f):
        return json.dump(self.data._asdict(), f, cls=JsonEncoder, indent=2)


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
        self.exp_id = uuid.uuid4().__str__().replace('-', '_')
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
        self.mode = None
        self.policy_net = DQN(env).to(self.args.device)
        self.target_net = DQN(env).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon_step = (self.args.eps - self.args.eps_min) / self.args.eps_decay_window

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1.5e-4, eps=0.001)

        # Compute Huber loss
        self.loss = F.smooth_l1_loss  #

        # todo: Bug fix on pytorch - https://github.com/pytorch/examples/issues/370
        # self.policy_net.share_memory()
        # self.target_net.share_memory()
        self.policy_net.train()
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.P = np.zeros(env.action_space.n, np.float32)

        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

        self.args.save_dir += f'/{self.exp_id}/'
        os.system(f"mkdir -p {self.args.save_dir}")
        self.meta = MetaData(fp=open(os.path.join(self.args.save_dir, 'result.csv'), 'w'), args=self.args)

        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            self.load_model()
            ###########################
            # YOUR IMPLEMENTATION HERE #

    def device(self, func):
        if self.args.device == "cpu":
            return func.cpu()
        elif self.args.device == "gpu":
            return func.cuda()
        else:
            raise Exception("Error")

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

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        with torch.no_grad():
            self.P.fill(self.cur_eps / self.nA)
            q, argq = self.policy_net(Variable(self.reset(observation))).data.cpu().max(1)
            self.P[argq[0].item()] += 1 - self.cur_eps
            action = torch.tensor([np.random.choice(self.A, p=self.P)])
        ###########################
        return action, q

    def select_action(self, state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.t / EPS_DECAY)
        self.t += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1), self.policy_net(state).max(1)[1]
        else:
            return torch.tensor([[random.randrange(self.nA)]], device=self.args.device, dtype=torch.long), \
                   self.policy_net(state).max(1)[1]

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
        batch = random.sample(self.memory, batch_size)
        ###########################
        # return random.sample(self.memory, batch_size)
        return map(lambda x: Variable(torch.cat(x, 0)), zip(*batch))

    def optimize_model(self):
        if len(self.memory) < self.args.mem_init_size:
            return 0
        self.mode = "Explore"
        bs, ba, bns, br, bdone = self.replay_buffer(self.args.batch_size)
        bq = self.policy_net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
        bnq = self.target_net(bns).detach().max(1)[0].squeeze(0) * self.args.gamma * (1 - bdone)

        # Compute Huber loss
        loss = self.loss(bq, br + bnq)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.cpu().detach().numpy()

    def reset(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[1] == 4:
            return state
        return torch.reshape(state, [1, 84, 84, 4]).permute(0, 3, 1, 2)

    def save_model(self, i_episode):
        if i_episode % self.args.save_freq == 0:
            model_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.th')
            meta_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.meta')
            print(f"Saving model at {model_file}")
            with open(model_file, 'wb') as f:
                torch.save(self.policy_net, f)
            with open(meta_file, 'w') as f:
                self.meta.dump(f)

    def collect_garbage(self, i_episode):
        if i_episode % self.args.gc_freq == 0:
            print("Executing garbage collector . . .")
            gc.collect()

    def load_model(self):
        print(f"Restoring model from {self.args.load_dir} . . . ")
        self.policy_net = torch.load(self.args.load_dir,
                                     map_location=torch.device(self.args.device)).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if not self.args.test_dqn:
            self.meta.load(open( self.args.load_dir.replace('.th', '.meta')))
            self.eps = self.meta.data.eps
            self.t = self.meta.data.step
        else:
            self.cur_eps = 0.01
        print(f"Model successfully restored.")


    def train(self):
            """
            Implement your training algorithm here
            """
            ###########################
            # YOUR IMPLEMENTATION HERE #

            self.eps = max(self.args.eps, self.args.eps_min)
            self.eps_delta = (self.eps - self.args.eps_min) / self.args.eps_decay_window
            self.t = 1
            self.eps = self.args.eps
            self.mode = "Random"
            train_start = time.time()
            if not self.args.load_dir == '':
                self.load_model()
            for i_episode in range(1, self.args.max_episodes + 1):
                # Initialize the environment and state
                start_time = time.time()
                state = self.reset(self.env.reset())
                self.R[i_episode % self.args.window] = 0
                self.L[i_episode % self.args.window] = 0
                self.M[i_episode % self.args.window] = -1e9
                self.ep_len = 0
                done = False

                self.collect_garbage(i_episode)
                self.save_model(i_episode)

                while not done:
                    # Update the target network, copying all weights and biases in DQN
                    if self.t % self.args.target_update == 0:
                        print("Updating target network . . .")
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    # Select and perform an action
                    self.cur_eps = max(self.args.eps_min, self.eps - self.eps_delta * self.t)
                    if self.cur_eps == self.args.eps_min:
                        self.mode = 'Exploit'
                    action, q = self.make_action(state)
                    next_state, reward, done, _ = self.env.step(action.item())
                    next_state = self.reset(next_state)
                    reward = torch.tensor([reward], device=self.args.device)
                    # Store the transition in memory
                    self.push(state, torch.tensor([int(action)]), next_state, reward,
                              torch.tensor([done], dtype=torch.float32))

                    self.R[i_episode % self.args.window] += reward
                    self.M[i_episode % self.args.window] = max(self.M[i_episode % self.args.window], q[0].item())

                    self.t += 1
                    self.ep_len += 1

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the target network)
                    if self.ep_len % self.args.learn_freq == 0:
                        loss = self.optimize_model()
                        self.L[i_episode % self.args.window] += loss

                self.meta.update(i_episode, self.t, time.time() - start_time, time.time() - train_start,
                                 self.ep_len, len(self.memory), self.cur_eps,
                                 self.R[i_episode % self.args.window], np.mean(self.R),
                                 self.M[i_episode % self.args.window], np.mean(self.M),
                                 self.L[i_episode % self.args.window], np.mean(self.L),
                                 self.mode)

            ###########################
