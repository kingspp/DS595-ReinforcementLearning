import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    ############################
    # YOUR IMPLEMENTATION HERE #
    # Declare an error array of size nS
    error = np.ones(nS) / nA
    # Policy Evaluation - Terminates on - max |value_function(s) - prev_value_function(s)| < tol
    while max(error) > tol:
        # Iterate for number of possible states
        for state in range(nS):
            # Initialize Value for current state
            value = 0
            # Iterate for all possible actions in given policy[state]
            for action, action_probability in enumerate(policy[state]):
                # Iterate for the possible states, given current action
                for probability, next_state, reward, done in P[state][action]:
                    # Value Function Calculation
                    value += action_probability * probability * (reward + gamma * (value_function[next_state]))
            # Store error for every given state. Abs is applied due to use of max operation in the while loop
            error[state] = np.abs(value_function[state] - value)
            # Update current state value in value function.
            value_function[state] = value
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA

    ############################
    # YOUR IMPLEMENTATION HERE #

    def get_values_for_actions_for_state(policy, nA, state, value, gamma):
        """
        The function calculates the value for n actions, given a state
        Args:
            policy : Policy
            nA : Next Action
            state: The state to consider (int)
            value: The value to use as an estimator, Vector of length env.nS
            gamma: Discount Factor

        Returns:
            Vector of length nA having values for each action.
        """
        # Initialize value array with same length as number of actions
        values = np.zeros(nA)
        # Iterate through actions
        for action in range(nA):
            # Given an action, iterate to possible next states
            for prob, next_state, reward, done in policy[state][action]:
                # Calculate the value using Bellman equation
                values[action] += prob * (reward + gamma * value[next_state])
        return values

    # Initialize policy stable to be False
    policy_stable = False
    # Loops for every stable state of the policy
    while not policy_stable:
        # Iterate for each state in states
        for state in range(nS):
            # Choose the best action from the current policy given a state
            best_action_from_new_policy = np.argmax(new_policy[state])
            # Find the best using value function
            best_action = np.argmax(get_values_for_actions_for_state(P, nA, state, value_from_policy, gamma))
            # If best action from the new policy is  same as the best action from value function, then
            # policy is stable. Exit the loop on finding a stable policy
            if best_action_from_new_policy == best_action:
                policy_stable = True
            # Convert 1d action array to a binary matrix representation.
            new_policy[state] = np.eye(nA)[best_action]
    ############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        # Calculate the value function of a Policy using Policy Evaluation
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        # Run Policy improvement using the given policy and value function for that policy
        new_policy = policy_improvement(P, nS, nA, V, gamma)
        # If new_policy is not changed from policy, then new_policy is stable. Exit the loop
        if (new_policy == policy).all():
            break
        # Update old policy to new policy
        policy = new_policy.copy()
    ############################
    return new_policy, V


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    V_new = V.copy()

    ############################
    # YOUR IMPLEMENTATION HERE #

    def get_values_for_actions_for_state(policy, nA, state, value, gamma):

        """
        The function calculates the value for n actions, given a state
        Args:
            policy : Policy
            nA : Next Action
            state: The state to consider (int)
            value: The value to use as an estimator, Vector of length env.nS
            gamma: Discount Factor

        Returns:
            Vector of length nA having values for each action.
        """

        # Initialize value array with same length as number of actions
        values = np.zeros(nA)
        # Iterate through actions
        for action in range(nA):
            # Given an action, iterate to possible next states
            for prob, next_state, reward, done in policy[state][action]:
                # Calculate the value using Bellman equation
                values[action] += prob * (reward + gamma * value[next_state])
        return values
    # Declare an error array with size of states
    error = np.ones(nS) / nA
    # Value Iteration - Terminates on - max |value_function(s) - prev_value_function(s)| < tol
    while max(error) > tol:
        # Iterate through each state
        for state in range(nS):
            # Fetch state index of previous value function
            Vs = V_new[state]
            # Get new best values for actions
            V_new[state] = max(get_values_for_actions_for_state(P, nA, state, V_new, gamma))
            # Calculate error for the state
            error[state] = abs(V_new[state] - Vs)

    # Given new value function, create a new policy.
    policy_new = policy_improvement(P, nS, nA, V_new, gamma)
    ############################
    return policy_new, V_new


def render_single(env, policy, render=False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game.
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset()  # initialize the episode
        print(f'Episode: {_}')
        done = False
        while not done:
            if render:
                env.render()  # render the game
            ############################
            # YOUR IMPLEMENTATION HERE #
            # Fetch the best action from the policy given the current observation
            action = np.argmax(policy[ob, :])
            # Send the action as a step
            ob, reward, done, info = env.step(action)
            # Accumulate the rewards
            total_rewards += reward
    return total_rewards
