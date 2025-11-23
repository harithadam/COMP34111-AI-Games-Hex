from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

import math
import random
import copy


class HXMinimaxAgent(AgentBase):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.max_depth = 2  # increase if time allows

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        legal_moves = self.get_legal_moves(board)
        
        # Handle swap rule on turn 2
        if turn == 2 and not self.has_swapped(board):
            legal_moves.append(Move(-1, -1))  # Add swap move

        # Shuffle moves to diversify play
        random.shuffle(legal_moves)

        best_score = -math.inf
        best_move = legal_moves[0]  # default to first move
        alpha, beta = -math.inf, math.inf

        for move in legal_moves:
            if move.is_swap():
                # For swap move, simulate the board state after swapping
                score = self.evaluate_swap_position(board)
            else:
                # simulate move
                board.set_tile_colour(move.x, move.y, self.colour)
                score = self.minimax(board, depth=self.max_depth - 1,
                                   alpha=alpha, beta=beta, maximizing=False)
                # undo move
                board.set_tile_colour(move.x, move.y, None)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move

