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
from dueling_dqn_model import DuelingDQN
from crnn_model import CrnnDQN
import time
from torch.autograd import Variable
import json
import uuid
from humanize import naturaltime

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


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
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.episode_template = namedtuple('EpisodeData',
                                           (
                                               "episode", "step", "time", "time_elapsed", "ep_len", "buffer_len",
                                               "epsilon",
                                               "reward", "avg_reward", "max_q", "max_avg_q", "loss", "avg_loss", "mode",
                                               "lr"))
        self.step_template = namedtuple('StepData', ("step", "epsilon", "reward", "max_q", "loss", "lr"))
        self.fp = fp
        self.episode_data = None
        self.step_data = None
        self.args = args
        if self.args.tb_summary:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter('/'.join(self.fp.name.split('/')[:-1]) + '/tb_logs/')

    def update_step(self, *args):
        self.step_data = self.step_template(*args)
        if self.args.tb_summary:
            self.writer.add_scalar('step/epsilon', self.step_data.epsilon, self.step_data.step)
            self.writer.add_scalar('step/learning_rate', self.step_data.lr, self.step_data.step)
            self.writer.add_scalar('step/reward', self.step_data.reward, self.step_data.step)
            self.writer.add_scalar('step/max_q', self.step_data.max_q, self.step_data.step)
            self.writer.add_scalar('step/loss', self.step_data.loss, self.step_data.step)

    def update_episode(self, *args):
        """
        Update metadata
        :param args: args
        """
        self.episode_data = self.episode_template(*args)
        if self.episode_data.episode % self.args.disp_freq == 0:
            print(
                f"E: {self.episode_data.episode} | M: {self.episode_data.buffer_len} |  Step: {self.episode_data.step} "
                f"| T: {self.episode_data.time:.2f} | Len: {self.episode_data.ep_len} | EPS: {self.episode_data.epsilon:.5f} "
                f"| LR: {self.episode_data.lr:.7f} | R: {self.episode_data.reward} | AR: {self.episode_data.avg_reward:.3f} "
                f"| MAQ:{self.episode_data.max_avg_q:.2f} "
                f"| L: {self.episode_data.loss:.2f} | AL: {self.episode_data.avg_loss:.4f} | Mode: {self.episode_data.mode} "
                f"| ET: {naturaltime(self.episode_data.time_elapsed)}")
        if self.args.tb_summary:
            self.writer.add_scalar('episode/epsilon', self.episode_data.epsilon, self.episode_data.episode)
            self.writer.add_scalar('episode/steps', self.episode_data.step, self.episode_data.episode)
            self.writer.add_scalar('episode/learning_rate', self.episode_data.lr, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_reward', self.episode_data.avg_reward, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_max_q', self.episode_data.max_avg_q, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_loss', self.episode_data.avg_loss, self.episode_data.episode)

        self.fp.write(self.episode_data._asdict().values().__str__().replace('odict_values([', '').replace('])', '\n'))

    def load(self, f):
        """
        Load Metadata
        :param f: File Pointer
        :return:
        """
        self.episode_data = self.episode_data(*json.load(f).values())

    def dump(self, f):
        """
        JSONify metadata
        :param f: file pointer
        """
        json.dump(self.episode_data._asdict(), f, cls=JsonEncoder, indent=2)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, args, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))
        self.args = args

    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(self.transition(*args))
        else:
            self.memory[self.pos] = self.transition(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return [*zip(*samples), indices,
                weights]  # [*map(lambda x: Variable(torch.cat(x, 0)).to(self.args.device), zip(*samples)), indices, weights]

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)


class ReplayBuffer(object):
    """ Facilitates memory replay. """

    def __init__(self, capacity, args):
        self.capacity = capacity
        self.memory = []
        self.idx = 0
        self.args = args
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = self.transition(*args)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, bsz):
        batch = random.sample(self.memory, bsz)
        return [*zip(*batch)]


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
        # Declare variables
        self.exp_id = uuid.uuid4().__str__().replace('-', '_')
        self.args = args
        self.env = env
        self.eps_threshold = None
        self.nA = env.action_space.n
        self.action_list = np.arange(self.nA)
        self.reward_list = deque(maxlen=args.window)  # np.zeros(args.window, np.float32)
        self.max_q_list = deque(maxlen=args.window)  # np.zeros(args.window, np.float32)
        self.loss_list = deque(maxlen=args.window)  # np.zeros(args.window, np.float32)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.cur_eps = self.args.eps
        self.t = 0
        self.ep_len = 0
        self.mode = None
        if self.args.use_pri_buffer:
            self.replay_buffer = NaivePrioritizedBuffer(capacity=self.args.capacity, args=self.args)
        else:
            self.replay_buffer = ReplayBuffer(capacity=self.args.capacity, args=self.args)
        self.position = 0

        self.args.save_dir += f'/{self.exp_id}/'
        os.system(f"mkdir -p {self.args.save_dir}")
        self.meta = MetaData(fp=open(os.path.join(self.args.save_dir, 'result.csv'), 'w'), args=self.args)
        self.eps_delta = (self.args.eps - self.args.eps_min) / self.args.eps_decay_window
        self.beta_by_frame = lambda frame_idx: min(1.0, args.pri_beta_start + frame_idx * (1.0 - args.pri_beta_start) / args.pri_beta_decay)

        # Create Policy and Target Networks
        if self.args.use_dueling:
            print("Using dueling dqn . . .")
            self.policy_net = DuelingDQN(env, self.args).to(self.args.device)
            self.target_net = DuelingDQN(env, self.args).to(self.args.device)
        elif self.args.use_crnn:
            print("Using dueling crnn . . .")
            self.policy_net = CrnnDQN(env).to(self.args.device)
            self.target_net = CrnnDQN(env).to(self.args.device)
        else:
            self.policy_net = DQN(env, self.args).to(self.args.device)
            self.target_net = DQN(env, self.args).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.lr, eps=self.args.optimizer_eps)
        if self.args.lr_scheduler:
            print("Enabling LR Decay . . .")
            self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.args.lr_decay)
        self.cur_lr = self.optimizer.param_groups[0]['lr']

        # Compute Huber loss
        self.loss = F.smooth_l1_loss

        # todo: Support for Multiprocessing. Bug in pytorch - https://github.com/pytorch/examples/issues/370
        self.policy_net.share_memory()
        self.target_net.share_memory()

        # Set defaults for networks
        self.policy_net.train()
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if args.test_dqn:
            # you can load your model here
            ###########################
            # YOUR IMPLEMENTATION HERE #
            print('loading trained model')
            self.load_model()

        if args.use_pri_buffer:
            print('Using priority buffer . . .')
        if args.use_double_dqn:
            print('Using double dqn . . .')

        if args.use_bnorm:
            print("Using batch normalization . . .")

        print("Arguments: \n", json.dumps(vars(self.args), indent=2), '\n')


    def init_game_setting(self):
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
            if self.args.test_dqn:
                q, argq = self.policy_net(Variable(self.channel_first(observation))).data.cpu().max(1)
                return self.action_list[argq]
            # Fill up probability list equal for all actions
            self.probability_list.fill(self.cur_eps / self.nA)
            # Fetch q from the model prediction
            q, argq = self.policy_net(Variable(self.channel_first(observation))).data.cpu().max(1)
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.cur_eps
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])

        ###########################
        return action.item(), q.item()

    def optimize_model(self):
        """
        Function to perform optimization on DL Network
        :return: Loss
        """
        # Return if initial buffer is not filled.
        if len(self.replay_buffer.memory) < self.args.mem_init_size:
            return 0
        if self.args.use_pri_buffer:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done, indices, weights = self.replay_buffer.sample(
                self.args.batch_size, beta=self.beta_by_frame(self.t))
        else:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.replay_buffer.sample(
                self.args.batch_size)
        batch_state = Variable(self.channel_first(torch.tensor(np.array(batch_state), dtype=torch.float32)))
        batch_action = Variable(torch.tensor(np.array(batch_action), dtype=torch.long))
        batch_next_state = Variable(self.channel_first(torch.tensor(np.array(batch_next_state), dtype=torch.float32)))
        batch_reward = Variable(torch.tensor(np.array(batch_reward), dtype=torch.float32))
        batch_done = Variable(torch.tensor(np.array(batch_done), dtype=torch.float32))
        policy_max_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        if self.args.use_double_dqn:
            policy_ns_max_q = self.policy_net(batch_next_state)
            next_q_value = self.target_net(batch_next_state).gather(1, torch.max(policy_ns_max_q, 1)[1].unsqueeze(
                1)).squeeze(1)
            target_max_q = next_q_value * self.args.gamma * (1 - batch_done)
        else:
            target_max_q = self.target_net(batch_next_state).detach().max(1)[0].squeeze(0) * self.args.gamma * (
                    1 - batch_done)
        # Compute Huber loss
        if self.args.use_pri_buffer:
            loss = (policy_max_q - (batch_reward + target_max_q.detach())).pow(2) * Variable(
                torch.tensor(weights, dtype=torch.float32))
            prios = loss + 1e-5
            loss = loss.mean()
        else:
            loss = self.loss(policy_max_q, batch_reward + target_max_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients between -1 and 1
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        if self.args.use_pri_buffer:
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        self.optimizer.step()
        return loss.cpu().detach().numpy()





    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        def train_fn():
            self.t = 1
            self.mode = "Random"
            train_start = time.time()
            if not self.args.load_dir == '':
                self.load_model()
            for i_episode in range(1, self.args.max_episodes + 1):
                # Initialize the environment and state
                start_time = time.time()
                state = self.env.reset()
                self.reward_list.append(0)
                self.loss_list.append(0)
                self.max_q_list.append(0)
                self.ep_len = 0
                done = False

                # Save Model
                self.save_model(i_episode)
                # Collect garbage
                self.collect_garbage(i_episode)

                # Run the game
                while not done:
                    # Update the target network, copying all weights and biases in DQN
                    if self.t % self.args.target_update == 0:
                        print("Updating target network . . .")
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    # Select and perform an action
                    self.cur_eps = max(self.args.eps_min, self.cur_eps - self.eps_delta)
                    if self.cur_eps == self.args.eps_min:
                        self.mode = 'Exploit'
                    else:
                        self.mode = "Explore"
                    action, q = self.make_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.reward_list[-1] += reward
                    self.max_q_list[-1] = max(self.max_q_list[-1], q)
                    # Store the transition in memory
                    self.replay_buffer.push(state, action, next_state, reward, done)
                    self.meta.update_step(self.t, self.cur_eps, self.reward_list[-1], self.max_q_list[-1],
                                          self.loss_list[-1], self.cur_lr)

                    # Increment step and Episode Length
                    self.t += 1
                    self.ep_len += 1

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the target network)
                    if self.ep_len % self.args.learn_freq == 0:
                        loss = self.optimize_model()
                        self.loss_list[-1] += loss
                self.loss_list[-1] /= self.ep_len

                # Decay Step:
                if self.args.lr_scheduler:
                    self.cur_lr = self.scheduler.get_lr()[0]
                    if i_episode % self.args.lr_decay_step == 0 and self.cur_lr > self.args.lr_min:
                        self.scheduler.step(i_episode)

                # Update meta
                self.meta.update_episode(i_episode, self.t, time.time() - start_time, time.time() - train_start,
                                         self.ep_len, len(self.replay_buffer.memory), self.cur_eps,
                                         self.reward_list[-1], np.mean(self.reward_list),
                                         self.max_q_list[-1], np.mean(self.max_q_list),
                                         self.loss_list[-1], np.mean(self.loss_list),
                                         self.mode, self.cur_lr)
        import multiprocessing as mp
        processes = []
        for rank in range(4):
            p = mp.Process(target=train_fn)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        ###########################
