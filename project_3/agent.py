"""

### NOTICE ###
DO NOT revise this file

"""
from environment import Environment
import torch
import os
import gc

class Agent(object):
    def __init__(self, env):
        self.env = env


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        This function must exist in agent

        Input:
            When running dqn:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        raise NotImplementedError("Subclasses should implement this!")


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        raise NotImplementedError("Subclasses should implement this!")

    def channel_first(self, state):
        """
        The action returned from the environment is nhwc, hence convert to nchw
        :param state: state
        :return: nchw state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[1] == 4:
            return state
        return torch.reshape(state, [-1, 84, 84, 4]).permute(0, 3, 1, 2)

    def save_model(self, i_episode):
        """
        Save Model based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.save_freq == 0:
            model_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.th')
            meta_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.meta')
            print(f"Saving model at {model_file}")
            with open(model_file, 'wb') as f:
                torch.save(self.policy_net, f)
            with open(meta_file, 'w') as f:
                self.meta.dump(f)


    def collect_garbage(self, i_episode):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.gc_freq == 0:
            print("Executing garbage collector . . .")
            gc.collect()

    def load_model(self):
        """
        Load Model
        :return:
        """
        print(f"Restoring model from {self.args.load_dir} . . . ")
        self.policy_net = torch.load(self.args.load_dir,
                                     map_location=torch.device(self.args.device)).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if not self.args.test_dqn and not self.args.restore_only_weights:
            print('Restoring steps and meta . . .')
            self.meta.load(open(self.args.load_dir.replace('.th', '.meta')))
            self.t = self.meta.episode_data.step
        print(f"Model successfully restored.")
