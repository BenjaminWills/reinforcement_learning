from typing import Tuple, List, Callable
from tqdm import tqdm

import numpy as np


class gamblers_problem:
    def __init__(
        self,
    ) -> None:
        pass

    def play_games(
        self,
        n_games: int,
        betting_strategy: Callable,
        heads_probability: float,
        initial_capital: int = None,
    ):
        """Plays multiple games of the gambling game.

        Parameters
        ----------
        n_games : int
            Number of games to play.
        betting_strategy : Callable
            The betting strategy that the gambler is using.
        heads_probability : float, optional
            Probability of the coin landing on heads, by default 0.4

        Returns
        -------
        List[Tuple[bool, List[Tuple[int, int]]]]
            A list of tuples containing the victory condition and the betting history.
        """
        victory_list = []
        bet_history_list = []
        for _ in tqdm(range(n_games)):
            victory, betting_history = self.play_game(
                betting_strategy=betting_strategy,
                heads_probability=heads_probability,
                initial_capital=initial_capital,
            )
            victory_list.append(victory)
            bet_history_list.append(betting_history)
        return victory_list, bet_history_list

    def play_game(
        self,
        betting_strategy: Callable,
        heads_probability: float = 0.4,
        initial_capital: int = None,
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """Plays the gambling game.

        Parameters
        ----------
        betting_strategy : Callable
            The betting strategy that the gambler is using.
        heads_probability : float, optional
            Probability of the coin landing on heads, by default 0.4
        initial_capital : int, optional
            Initial capital that the agent has., by default None

        Returns
        -------
        Tuple[bool,List[Tuple[int, int]]]
            A tuple containing the victory condition and the betting history.
        """
        # Define initial capital
        if initial_capital is None:
            capital = np.random.randint(1, 100)
        if initial_capital is not None:
            capital = initial_capital

        # Victory condition
        victory = False

        betting_history = []
        while True:
            bet = betting_strategy(capital)
            betting_history.append([capital, bet])

            # Update capital
            probability_sample = np.random.uniform(0, 1)
            if probability_sample < heads_probability:
                capital += bet
            else:
                capital -= bet

            if capital == 100:
                victory = True
                break

            if capital == 0:
                victory = False
                break

        return victory, betting_history
