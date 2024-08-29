import random
from typing import List, Dict, Tuple, Literal

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

POSSIBLE_ACTIONS = ["hit", "stick"]
REWARDS = [-1, 0, 1]
DISCOUNT_FACTOR = 1

deck = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
deck_values = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11 | 1,
}


def draw_card(deck: DECK) -> CARD:
    """
    Draws a card from the deck.

    Args:
        deck (DECK): The deck of cards.

    Returns:
        CARD: The drawn card.
    """
    return random.choice(deck)


def evaluate_hand(hand: HAND, deck_values: Dict[str, int]) -> int:
    """
    Evaluates the total value of a hand.

    Args:
        hand (HAND): The hand of cards.
        deck_values (Dict[str, int]): The dictionary mapping card values.

    Returns:
        int: The total value of the hand.
    """
    total = 0
    # Calculate the non-ace sum.
    for card in hand:
        if card != "A":
            total += deck_values[card]
    # Calculate the ace sum.
    for card in hand:
        if card == "A":
            if total <= 10:
                total += 11
            else:
                total += 1
    return total


def victory_condition(player_score: int, dealer_score: int) -> int:
    """
    Determines the victory condition based on player and dealer scores.

    Args:
        player_score (int): The score of the player.
        dealer_score (int): The score of the dealer.

    Returns:
        int: The victory condition (-1 for player loss, 0 for draw, 1 for player win).
    """
    if player_score > 21:
        # Player goes bust
        return -1
    elif dealer_score > 21:
        # Dealer goes bust
        return 1
    elif player_score == dealer_score:
        # Players score matches the dealers score
        return 0
    elif player_score > dealer_score:
        # Player has a higher score than the dealer
        return 1
    else:
        # Player has a lower score than the dealer
        return -1


def play_game_without_strategy(deck: DECK, deck_values: Dict[str, int]) -> int:
    """
    Plays a game of blackjack without using any strategy.

    Args:
        deck (DECK): The deck of cards.
        deck_values (Dict[str, int]): The dictionary mapping card values.

    Returns:
        int: The victory condition of the game (-1 for player loss, 0 for draw, 1 for player win).
    """
    # Deal the cards
    player_hand = [draw_card(deck) for _ in range(2)]
    dealer_hand = [draw_card(deck) for _ in range(2)]

    # Player's turn
    player_score = evaluate_hand(player_hand, deck_values)

    while player_score < 21:
        drawn_card = draw_card(deck)
        player_hand.append(drawn_card)
        player_score = evaluate_hand(player_hand, deck_values)

    # Dealer's turn
    dealer_score = evaluate_hand(dealer_hand, deck_values)
    while dealer_score <= 16:
        drawn_card = draw_card(deck)
        dealer_hand.append(drawn_card)
        dealer_score = evaluate_hand(dealer_hand, deck_values)

    return victory_condition(player_score, dealer_score)


def play_game_with_strategy(
    deck: DECK, deck_values: Dict[str, int], policy: POLICY
) -> Tuple[int, List[Tuple[STATE, ACTIONS]]]:
    """
    Plays a game of blackjack using a given strategy.

    Args:
        deck (DECK): The deck of cards.
        deck_values (Dict[str, int]): The dictionary mapping card values.
        policy (POLICY): The policy for choosing actions.

    Returns:
        Tuple[int, List[Tuple[STATE, ACTIONS]]]: The victory condition of the game and the history of state-action pairs.
    """
    # Deal the cards
    player_hand = [draw_card(deck) for _ in range(2)]
    dealer_hand = [draw_card(deck) for _ in range(2)]

    # Player's turn
    player_score = evaluate_hand(player_hand, deck_values)
    player_state_action_history = []

    while player_score < 21:
        state = (
            dealer_hand[0],
            player_score,
            "A" in player_hand and player_score <= 11,
        )
        # Sample from the hit distribution
        if policy(state) == "hit":
            player_state_action_history.append((state, "hit"))
            drawn_card = draw_card(deck)
            player_hand.append(drawn_card)
            player_score = evaluate_hand(player_hand, deck_values)
        else:
            player_state_action_history.append((state, "stick"))
            break

    # Dealer's turn
    dealer_score = evaluate_hand(dealer_hand, deck_values)
    while dealer_score <= 16:
        drawn_card = draw_card(deck)
        dealer_hand.append(drawn_card)
        dealer_score = evaluate_hand(dealer_hand, deck_values)

    return victory_condition(player_score, dealer_score), player_state_action_history
