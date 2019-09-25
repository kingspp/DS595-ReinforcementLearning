import gym
import numpy as np
import sys
from collections import defaultdict
from collections import Counter

env = gym.make('CliffWalking-v0')


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a.
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project_2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    props = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    props[best_action] += 1. - epsilon
    action = np.random.choice(np.arange(len(props)), p=props)
    ############################
    return action


def test_epsilon_greedy():
    '''epsilon_greedy (0 point)'''
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    state = 5

    actions = []
    for _ in range(10000):
        action = epsilon_greedy(Q, state, 4, epsilon=0.1)
        actions.append(action)

    assert np.allclose(1 - np.count_nonzero(actions) / 10000, 0.925, atol=0.02)


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a.
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """

    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

    # define decaying epsilon
    decay_epsilon = lambda e:  0.99*e

    # loop n_episodes
    for episode in range(n_episodes):
        # initialize the environment
        current_state = env.reset()
        # get an action from policy
        current_action = epsilon_greedy(Q, current_state, env.nA, epsilon)
        epsilon = decay_epsilon(epsilon)
        while True:
            # one step in the environment
            # return a new state, reward and done
            next_state, reward, done, _ = env.step(current_action)
            # get next action
            next_action = epsilon_greedy(Q, next_state, env.nA, epsilon)
            # TD update
            # td_target
            td_target = reward + gamma * Q[next_state][next_action]
            # td_error
            td_error = td_target - Q[current_state][current_action]
            # new Q
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * td_error
            if done:
                break
            # update state
            current_state = next_state
            # update action
            current_action = next_action

    ############################
    return Q


def test_sarsa():
    '''SARSA (25 points)'''
    test_policy = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])

    Q_s = sarsa(env, n_episodes=50000, gamma=1.0, alpha=0.01, epsilon=0.1)
    policy_q = np.array([np.argmax(Q_s[key]) if key in Q_s else -1 for key
                         in np.arange(48)]).reshape((4, 12))
    print(policy_q)
    assert np.allclose(policy_q.shape, (4, 12))
    assert np.allclose(policy_q[2:, ], test_policy)


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a.
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop n_episodes
    for episode in range(n_episodes):

        # initialize the environment
        current_state = env.reset()

        while True:
            # one step in the environment
            # get an action from policy
            current_action = epsilon_greedy(Q, current_state, env.nA, epsilon)
            # return a new state, reward and done
            next_state, reward, done, _ = env.step(current_action)
            # TD update
            best_next_action = np.argmax(Q[next_state])
            # td_target with best Q
            td_target = reward + gamma * Q[next_state][best_next_action]
            # td_error
            td_error = td_target - Q[current_state][current_action]
            # new Q
            Q[current_state][current_action] = Q[current_state][current_action] + alpha * td_error
            if done:
                break
            # update state
            current_state = next_state
            ############################
    return Q


def test_q_learning():
    '''Q_learning (25 points)'''
    Q_q = q_learning(env, n_episodes=10000, gamma=1.0, alpha=0.01, epsilon=0.1)
    policy_q = np.array([np.argmax(Q_q[key]) if key in Q_q else -1 for key
                         in np.arange(48)]).reshape((4, 12))
    test_policy = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])

    print(policy_q)
    assert np.allclose(policy_q.shape, (4, 12))
    assert np.allclose(policy_q[2:, ], test_policy)
