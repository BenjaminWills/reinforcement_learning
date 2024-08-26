# Random walk with 7 states. At each state we have a 50% probability of moving to the next, we begin at the middle state.
from typing import Dict, List, Callable, Tuple
from tqdm import tqdm
from copy import copy

PROBABILITY_UP = float
STATE = int
POLICY = Dict[STATE, PROBABILITY_UP]
VALUE_FUNCTION = Dict[STATE, float]


def initialise_value_function() -> VALUE_FUNCTION:
    """Initialise the value function for the random walk.

    Returns
    -------
    VALUE_FUNCTION
        The value function for the random walk.
    """
    return {state: 0.5 for state in range(-2, 3)}


def initialise_policy() -> POLICY:
    """Initialise the policy for the random walk.

    Returns
    -------
    POLICY
        The policy for the random walk, gives the probability of going up or down.
    """
    return {state: 0.5 for state in range(-3, 4)}


def update_value_function(
    alpha: float,
    episode_reward: int,
    value_function: VALUE_FUNCTION,
    state_history: List[int],
) -> VALUE_FUNCTION:
    """Update the value function based on the episode reward.

    Parameters
    ----------
    alpha : float
        The step size parameter - low = low variance slow convergence, high is the opposite.
            Note: this parameter is fixed and must be between 0 and 1.
    episode_reward : int
        The reward for the episode.
    value_function : VALUE_FUNCTION
        The value function to update
    state_history : List[int]
        The history of the states visited in the episode.

    Returns
    -------
    VALUE_FUNCTION
        The updated value function.
    """
    updated_value_function = copy(value_function)
    for state in state_history:
        updated_value_function[state] = updated_value_function[state] + alpha * (
            episode_reward - updated_value_function[state]
        )
    return updated_value_function


def fixed_alpha_monte_carlo_evaluation(
    alpha: float,
    episodes: int,
    random_walk_function: Callable,
) -> Tuple[VALUE_FUNCTION, List[VALUE_FUNCTION]]:
    """The fixed alpha monte carlo evaluation algorithm.

    Parameters
    ----------
    alpha : float
        The step size parameter - low = low variance slow convergence, high is the opposite.
            Note: this parameter is fixed and must be between 0 and 1.
    episodes : int
        The number of episodes to run the algorithm for. Remember that monte-carlo methods will sample
        from a game distribution.
    random_walk_function : Callable
        The function that will be used to generate the random walk.

    Returns
    -------
    Tuple[VALUE_FUNCTION, List[VALUE_FUNCTION]]
        The value function and the history of the value function for each episode.
    """
    # The idea is that for each episode we will update the value function

    policy = initialise_policy()
    value_function = initialise_value_function()
    value_function_history = []
    for episode in tqdm(range(episodes)):
        episode_result, state_history = random_walk_function(policy)
        episode_return = int(episode_result)
        value_function = update_value_function(
            alpha, episode_return, value_function, state_history
        )

        if episode % 1000 == 0:
            value_function_history.append(value_function)

    return value_function, value_function_history
