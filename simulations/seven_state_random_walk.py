# Random walk with 7 states. At each state we have a 50% probability of moving to the next, we begin at the middle state.
import random
from typing import Dict, Tuple, List

PROBABILITY_UP = float
STATE = int
POLICY = Dict[STATE, PROBABILITY_UP]
VICTORY_CONDITION = bool


def random_walk(policy: POLICY) -> Tuple[VICTORY_CONDITION, List[STATE]]:
    """Simulate a random walk with 7 states.

    Parameters
    ----------
    policy : POLICY
        The policy to follow, gives probabiliy of going up or down.

    Returns
    -------
    Tuple[VICTORY_CONDITION, List[STATE]]
        A tuple containing the victory condition and the state history.
    """
    current_state = 0

    terminal_state_win = 3
    terminal_state_lose = -3
    state_history = []

    while True:
        if current_state == terminal_state_win:
            return True, state_history
        elif current_state == terminal_state_lose:
            return False, state_history
        state_history.append(current_state)
        # Sample from the policy distribution
        if random.random() < policy[current_state]:
            current_state += 1
        else:
            current_state -= 1
