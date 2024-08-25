from typing import Dict, Tuple, List, Literal
from tqdm import tqdm
import numpy as np

# Define data types for the Gambler's Problem
BET = int
STATE = int
POLICY = Dict[STATE, BET]
STATE_VALUE = Dict[STATE, float]
ACTION_VALUE = Dict[Tuple[STATE, BET], float]

# Define discount value for reward function
DISCOUNT_VALUE = 1.0


def initialise_state_value_function() -> STATE_VALUE:
    """Inialise the state value function.

    Returns
    -------
    STATE_VALUE
        The state value function for each possible state
    """
    # Value function looks like a dictionary of pairs of capitals and bets,
    # it looks like this: {(capital, bet): value}, so for each bet
    state_value_function = {}
    for capital in range(0, 101):
        state_value_function[capital] = 0

    return state_value_function


def initialise_policy() -> POLICY:
    """Initialise the policy for the gambler.

    Returns
    -------
    POLICY
        The policy for the gambler, which is a dictionary of states and bets.
    """

    # The policy is a dictionary of actions and their probabilities.
    policy = {
        state: np.random.randint(0, min(state, 100 - state) + 1)
        for state in range(1, 101)
    }
    return policy


def calculate_actions(state: int) -> List[BET]:
    """Calcualtes all possible actions given a state.

    Parameters
    ----------
    state : int
        The current state of the gambler.

    Returns
    -------
    List[BET]
        A list of all possible actions (bets) that the gambler can make.
    """
    # You can either bet what you have (given that it's less than 100, or you can bet the maximum amount that will get you to 100).
    # E.g you can't bet 60 if you already have 50, but you can bet 50.
    possible_actions = list(range(1, min(state, 100 - state) + 1))
    return possible_actions


def indicator_function(x: float, target: float) -> Literal[0, 1]:
    """The indicator function is a function that returns 1 if x is equal to the target, otherwise it returns 0.

    Parameters
    ----------
    x : float
        A number that we are checking against the target.
    target : float
        The target that we are checking against.

    Returns
    -------
    Literal[0,1]
        0 if x is not equal to the target, 1 if it is equal to the target.
    """
    return 1 if x == target else 0


def normalise_state_value_function(state_value_function: STATE_VALUE) -> STATE_VALUE:
    """Normalise the state value function by dividing all values by the maximum value in the state value function.

    Parameters
    ----------
    state_value_function : STATE_VALUE
        The state value function that we are normalising.

    Returns
    -------
    STATE_VALUE
        The normalised state value function.
    """
    # We normalise the state value function by dividing all values by the maximum value in the state value function.
    max_value = max(state_value_function.values())
    normalised_state_value_function = {
        state: value / max_value for state, value in state_value_function.items()
    }
    return normalised_state_value_function


def generate_action_value_function(
    state_value_function: STATE_VALUE, heads_probability: float
) -> ACTION_VALUE:
    """Generate the action value function given the state value function.

    Parameters
    ----------
    state_value_function : STATE_VALUE
        The state value function that we are using to generate the action value function.

    Returns
    -------
    ACTION_VALUE
        The action value function that we have generated.
    """
    action_value_equation = {}
    for capital in state_value_function:
        for bet in range(0, min(capital, 100 - capital) + 1):
            action_value_equation[(capital, bet)] = action_bellman_equation(
                capital, bet, state_value_function, heads_probability
            )
    return action_value_equation


def evaluate_policy(
    policy: POLICY,
    state_value_function: STATE_VALUE,
    action_value_function: ACTION_VALUE,
    heads_probability: float,
) -> STATE_VALUE:
    """Evaluate the policy given the state value function and the action value function.

    Parameters
    ----------
    policy : POLICY
        The policy that we are evaluating.
    state_value_function : STATE_VALUE
        The state value function that we are using to evaluate the policy.
    action_value_function : ACTION_VALUE
        The action value function that we are using to evaluate the policy.

    Returns
    -------
    STATE_VALUE
        The updated state value function given the policy.
    """
    while True:
        # We evaluate the state value function using the bellman equation.
        updated_state_value_function = {}
        for capital in state_value_function:
            bet = policy.get(capital, 0)
            updated_state_value_function[capital] = action_value_function[
                (capital, bet)
            ]

        if updated_state_value_function == state_value_function:
            break
        else:
            state_value_function = updated_state_value_function
            action_value_function = generate_action_value_function(
                state_value_function, heads_probability
            )
        return updated_state_value_function, action_value_function


def improve_policy(action_value_function: ACTION_VALUE) -> POLICY:
    """Given a value function of the form:

    q(s,a) = value, which looks like this in code:

    ```
    {
        (capital, bet): value,
        ...
    }
    ```

    So what we need to do is filter by a specific capital in the value function and find
    the bet that has the highest action value.

    Parameters
    ----------
    state : int
        The capital that the player is currently at.
    value_function : Dict[Tuple[int,int], float]
        The value function that we are using to improve the policy.

    Returns
    -------
    POLICY
        The improved policy that we have found.
    """
    improved_policy = {}
    for state in range(1, 100):
        # Calculate all possible actions from this state
        possible_actions = calculate_actions(state)

        # Initialise biggest value and best action variables
        biggest_value = -1
        best_action = None

        # We then look through all actions and find the one with the highest value
        for action in possible_actions:
            # Get the value of the state-action pair
            value = action_value_function[(state, action)]
            if value > biggest_value:
                # If the value is bigger than the biggest value, we update the biggest value and best action
                biggest_value = value
                best_action = action
        # We then update the policy with the best action for this state given our current action value function
        improved_policy[state] = best_action
    # This policy is guaranteed to be better than the previous policy or equivalent to it.
    return improved_policy


def policy_iteration(
    heads_probability: float,
) -> Tuple[POLICY, STATE_VALUE, ACTION_VALUE]:
    """The policy iteration algorithm.

    Returns
    -------
    Tuple[POLICY, STATE_VALUE, ACTION_VALUE]
        The optimal policy, state value function and action value function.
    """
    # Initialise the value function and policy
    state_value_function = initialise_state_value_function()
    policy = initialise_policy()

    MAX_ITERATIONS = 100_000

    # We then iterate through the policy evaluation and policy improvement steps until the policy converges
    for _ in tqdm(range(MAX_ITERATIONS)):
        # Generate the action value function from the value function using the bellman equation
        action_value_function = generate_action_value_function(
            state_value_function, heads_probability
        )

        # Evaluate the policy
        state_value_function, action_value_function = evaluate_policy(
            policy, state_value_function, action_value_function, heads_probability
        )

        # Improve the policy
        new_policy = improve_policy(action_value_function)

        # If the new policy is the same as the old policy, we have converged and we can return the policy
        if new_policy == policy:
            print(f"Converged to optimal policy in {_:,} iterations.")
            break

        # Otherwise, we update the policy and continue the loop
        policy = new_policy

    return (policy, state_value_function, action_value_function)


def action_bellman_equation(
    state: STATE,
    action: BET,
    state_value_function: STATE_VALUE,
    heads_probability: float,
) -> float:
    """The bellman equation that links the state value function to the action value function.

    Parameters
    ----------
    state : STATE
        The current state of the gambler.
    action : BET
        The current action that the gambler is taking.
    state_value_function : STATE_VALUE
        The state value function that we are using to calculate the action value function.

    Returns
    -------
    float
        The value of the action value function given the state, action and state value function.

    """

    # This is the bellman equation for the action value function.
    return (1 - heads_probability) * DISCOUNT_VALUE * state_value_function[
        state - action
    ] + heads_probability * (
        indicator_function(state + action, 100) + state_value_function[state + action]
    )


class Gambler_policy_iteration:
    def __init__(self, heads_probability: float = 0.4) -> None:
        self.policy, self.value_function, self.action_value_function = policy_iteration(
            heads_probability
        )
