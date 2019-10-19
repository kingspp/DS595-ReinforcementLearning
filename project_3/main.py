import argparse
from test import test
from environment import Environment
import torch


def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--max_episodes', type=int, default=10000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=0.99)
    parser.add_argument('--eps_decay_window', type=int, default=1000000)
    parser.add_argument('--eps_min', type=float, default=0.1)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--capacity', type=int, default=50000)
    parser.add_argument('--mem_init_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target_update', type=int, default=10000)
    parser.add_argument('--learn_freq', type=int, default=4)
    parser.add_argument('--gc_freq', type=int, default=1000)
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--disp_freq', type=int, default=100)
    parser.add_argument('--optimizer_eps', type=int, default=0.001)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--use_pri_buffer', type=bool, default=True)
    parser.add_argument('--use_double_dqn', type=bool, default=True)
    parser.add_argument('--use_dueling', type=bool, default=True)
    parser.add_argument('--lr', type=int, default=1e-2)
    parser.add_argument("--lr_scheduler", type=bool, default=True)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_decay", type=float, default=0.9999)
    parser.add_argument("--lr_decay_step", type=float, default=4)


    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if args.device == "cuda" else 'torch.FloatTensor')
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
