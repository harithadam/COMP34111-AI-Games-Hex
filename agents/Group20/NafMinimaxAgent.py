# Authored by: Naf

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Game import logger

import math
import random
import copy
import time

INF = 10**9

class NafMinimaxAgent(AgentBase):
    """
    Beam-search minimax Hex agent
    - Uses an evaluation function based on approximate connection distance
    - Alpha-beta pruning + move ordering
    - Optional beam restrictions at each node (beam_width > 0)
    """
    _board_size: int = 11

    def __init__(self, colour: Colour, max_depth: int = 3, beam_width: int = 8, time_limit_seconds: float = 1.0):
        super().__init__(colour)
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.time_limit = time_limit_seconds
        self._start_time = 0.0
        self._my_colour = colour

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Choose a move using iterative deepening minimax / beam search
        """
        self._start_time = time.time()
        legal_moves = self._generate_moves(board)

        # If there is no legal move
        if not legal_moves:
            logger.warning("No log moves; returning (0,0) as fallback.")
            return Move(0,0)
        
        # Fallback : if only one move, play it
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = legal_moves[0]
        best_val = -INF

        # Iterative deepening: try depth 1, 2, ..., max_depth
        for depth in range(1, self.max_depth + 1):
            if self._time_up():
                break

            value, move = self._search_root(board, depth)
            if self._time_up():
                break

            if move is not None:
                best_move = move
                best_val = value

            logger.debug(f"Depth {depth}: best value {best_val}, move {best_move.x},{best_move.y}")

        return best_move
    
    def _search_root(self, board: Board, depth: int) -> tuple[float, Move | None]:
        """
        Root search
        
        :param self: Description
        :param board: Description
        :type board: Board
        :param depth: Description
        :type depth: int
        :return: Description
        :rtype: tuple[float, Move | None]
        """
        legal_moves = self._generate_moves(board)

        # Order moves by heuristic so alpha-beta prunes better
        ordered_moves = self._order_moves(board, legal_moves, self._my_colour)
        

