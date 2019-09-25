import sys
import gym
import numpy as np
from collections import defaultdict


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation

    # action
    if observation[0] >= 20:
        action = 0
    else:
        action = 1

    ############################
    return action


env = gym.make('Blackjack-v0')


# # ---------------------------------------------------------------
# # def t_initial_policy():
# #     '''initial_policy (2 points)'''
# #     state1 = (21, 10, True)
# #     action1 = initial_policy(state1)
# #     state2 = (18, 5, True)
# #     action2 = initial_policy(state2)
# #
# #     assert np.allclose(action1, 0)
# #     assert np.allclose(action2, 1)
# #
# # # ---------------------------------------------------------------
# #
# #
# # t_initial_policy()


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    ############################
    # YOUR IMPLEMENTATION HERE #

    ############################

    # loop each episode
    for i_episode in range(n_episodes):
        # initialize the episode
        state = env.reset()
        # generate empty episode
        episodes = []
        # loop until episode generation is done
        while True:
            # select an action
            action = policy(state)
            # return a reward and new state
            next_state, reward, done, _ = env.step(action)
            # append state, action, reward to episode
            episodes.append((state, action, reward))
            if done:
                break
            # update state to new state
            state = next_state

        states_set = set([x[0] for x in episodes])
        # loop each state
        for i, state in enumerate(states_set):
            # first occurence of the observation
            # return the first index of state
            idx = episodes.index([episode for episode in episodes if episode[0] == state][0])

            # sum up all rewards with discount_factor since the first visit
            Q = sum([episode[2] * gamma ** i for episode in episodes[idx:]])

            # calculate average return for this state over all sampled episodes
            returns_sum[state] += Q
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    ############################
    return V


# ---------------------------------------------------------------
def test_mc_prediction():
    '''mc_prediction (20 points)'''
    V_500k = mc_prediction(initial_policy, env, n_episodes=500000, gamma=1.0)

    boundaries1 = [(18, 4, False), (18, 6, False), (18, 8, False)]
    boundaries2 = [(18, 4, True), (18, 6, True), (18, 8, True)]
    boundaries3 = [(20, 4, False), (20, 6, False), (20, 8, False), (20, 4, True), (20, 6, True), (20, 8, True)]
    assert len(V_500k) == 280
    for b in boundaries1:
        assert np.allclose(V_500k[b], -0.7, atol=0.05)
    for b in boundaries2:
        assert np.allclose(V_500k[b], -0.4, atol=0.1)
    for b in boundaries3:
        assert V_500k[b] > 0.6


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
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    props = np.ones(nA, dtype=float) * epsilon / nA
    best_action = np.argmax(Q[state])
    props[best_action] += 1. - epsilon
    action = np.random.choice(np.arange(len(props)), p=props)
    ############################
    return action


# ---------------------------------------------------------------
def test_epsilon_greedy():
    '''epsilon_greedy (8 points)'''
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    state = (14, 7, True)

    actions = []
    for _ in range(10000):
        action = epsilon_greedy(Q, state, 4, epsilon=0.1)
        actions.append(action)

    assert np.allclose(1 - np.count_nonzero(actions) / 10000, 0.925, atol=0.02)


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = 1-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    ############################
    # YOUR IMPLEMENTATION HERE #

    # define decaying epsilon
    decay_epsilon = lambda e: e - (0.1 / n_episodes)
    # generate empty episode
    episodes = []
    # calculate average return for this state over all sampled episodes
    for i_episode in range(1, n_episodes + 1):
        # initialize the episode
        state = env.reset()
        epsilon = decay_epsilon(epsilon)
        # loop until one episode generation is done
        while True:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, state, 2, epsilon)
            # return a reward and new state
            next_state, reward, done, _ = env.step(action)
            # append state, action, reward to episode
            episodes.append((state, action, reward))
            if done:
                break
            # update state to new state
            state = next_state

        # find unique (state, action) pairs we've visited in this episode
        # each state should be a tuple so that we can use it as a dict key
        pairs = set([(episode[0], episode[1]) for episode in episodes])
        # loop each state,action pair
        for (state, action) in pairs:
            pair = (state, action)
            # find the first occurance of the (state, action) pair in the episode
            idx = episodes.index(
                [episode for episode in episodes if episode[0] == state and episode[1] == action][0])
            # sum up all rewards since the first occurance
            V = sum([reward[2] * gamma ** i for i, reward in enumerate(episodes[idx:])])
            returns_sum[pair] += V
            returns_count[pair] += 1.
            Q[state][action] = returns_sum[pair] / returns_count[pair]
    return Q


def test_mc_control_epsilon_greedy():
    '''mc_control_epsilon_greedy (20 points)'''
    boundaries_key = [(19, 10, True), (19, 4, True), (18, 7, True), (17, 9, True), (17, 5, True),
                      (17, 8, False), (17, 6, False), (15, 6, False), (14, 7, False)]
    boundaries_action = [0, 0, 0, 1, 1, 0, 0, 0, 1]

    count = 0
    for _ in range(2):
        Q_500k = mc_control_epsilon_greedy(env, n_episodes=1000000, gamma=1.0, epsilon=0.1)
        policy = dict((k, np.argmax(v)) for k, v in Q_500k.items())
        print([policy[key] for key in boundaries_key])
        if [policy[key] for key in boundaries_key] == boundaries_action:
            count += 1

    assert len(Q_500k) == 280
    assert count >= 1


test_mc_control_epsilon_greedy()
