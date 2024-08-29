import random

from typing import List, Dict, Tuple, Literal, Callable
from tqdm import tqdm

CARD = str
DECK = List[CARD]
HAND = List[CARD]

DEALERS_CARD = CARD
CURRENT_HAND_SUM = int
USEABLE_ACE = bool

STATE = Tuple[DEALERS_CARD, CURRENT_HAND_SUM, USEABLE_ACE]
ACTIONS = Tuple[Literal["hit"], Literal["stick"]]
HIT_PROBABILITY = float

POLICY = Dict[STATE, HIT_PROBABILITY]
ACTION_VALUE_FUNCTION = Dict[Tuple[STATE, ACTIONS], float]

POSSIBLE_ACTIONS = ["hit", "stick"]
REWARDS = [-1, 0, 1]
DISCOUNT_FACTOR = 1


def initialise_action_value_function(deck: DECK) -> ACTION_VALUE_FUNCTION:
    """Initialise the action value function. This is a function with a state and action of the following form:

    ((dealer_card, hand_sum, useable_ace), action)

    Parameters
    ----------
    deck : DECK
        The possible cards that can be drawn from the deck.

    Returns
    -------
    ACTION_VALUE_FUNCTION
        The action value function initialised to 0 for all state-action pairs.
    """
    # All possible states.
    # State[0] = Dealer's card, this can be one of 13 values.
    # State[1] = Current hand sum, this will be between 4 (two 2's) and 21.
    # State[2] = Useable ace, this can be one of 2 values: True or False.
    action_value_function = {}
    for dealer_card in deck:
        for hand_sum in range(4, 22):
            for useable_ace in [True, False]:
                for action in POSSIBLE_ACTIONS:
                    action_value_function[
                        ((dealer_card, hand_sum, useable_ace), action)
                    ] = 0
    return action_value_function


def initialise_epsilon_soft_policy(deck: DECK) -> POLICY:
    """Initialise the epsilon soft policy. This policy is a dictionary with the state as the key and the value as the probability of hitting.
    it is episilon soft as it is a policy that will generate any action with a non-zero probability.

    Parameters
    ----------
    deck : DECK
        The possible cards that can be drawn from the deck.

    Returns
    -------
    POLICY
        The epsilon soft policy.
    """
    # Define the policy to be the probability of hitting, this policy generates teh data
    epsilon_soft_policy = {}
    for dealer_card in deck:
        for hand_sum in range(4, 22):
            for useable_ace in [True, False]:
                epsilon_soft_policy[(dealer_card, hand_sum, useable_ace)] = (
                    random.choice(POSSIBLE_ACTIONS)
                )
    return epsilon_soft_policy


def epsilon_greedy_policy(
    state: STATE, action_value_function: ACTION_VALUE_FUNCTION, epsilon: float
) -> ACTIONS:
    """The epsilon greedy policy. This policy will select the action with the highest value with probability 1 - epsilon
    and a random exploritory action with probability epsilon.

    Parameters
    ----------
    state : STATE
        The current state of the environment.
    action_value_function : ACTION_VALUE_FUNCTION
        The action value function.
    epsilon : float
        The probability of selecting a random action.

    Returns
    -------
    ACTIONS
        The action to take. Either "hit" or "stick".
    """
    if random.random() < epsilon:
        return random.choice(POSSIBLE_ACTIONS)
    else:
        # Get the action with the highest value.
        hit_value = action_value_function[(state, "hit")]
        stick_value = action_value_function[(state, "stick")]
        if hit_value > stick_value:
            return "hit"
        else:
            return "stick"


def constant_alpha_montecarlo(
    epsilon: float,
    episodes: int,
    learning_rate: float,
    deck: DECK,
    deck_values: Dict[CARD, int],
    play_game: Callable,
) -> POLICY:
    """The constant alpha montecarlo algorithm. This algorithm will learn the optimal policy for the game of blackjack.

    Parameters
    ----------
    epsilon : float
        The probability of selecting a random action.
    episodes : int
        The number of episodes to run the algorithm for.
    learning_rate : float
        The learning rate of the algorithm.
    deck : DECK
        The possible cards that can be drawn from the deck.
    deck_values : Dict[CARD, int]
        The value of each card in the deck.
    play_game : Callable
        The function that will play a game of blackjack.

    Returns
    -------
    POLICY
        The optimal policy for the game of blackjack.
    """
    # Initialse values
    soft_policy = initialise_epsilon_soft_policy(deck)
    soft_policy_function = lambda state: soft_policy[state]
    action_value_function = initialise_action_value_function(deck)

    for episode in tqdm(range(episodes)):
        outcome, player_state_action_history = play_game(
            deck, deck_values, soft_policy_function
        )
        # in this case the reward is always the outcome.
        for player_turn, (state, action) in enumerate(player_state_action_history):
            action_value_function[(state, action)] += learning_rate * (
                outcome - action_value_function[(state, action)]
            )

        soft_policy_function = lambda state: epsilon_greedy_policy(
            state, action_value_function, epsilon
        )

    return soft_policy_function
