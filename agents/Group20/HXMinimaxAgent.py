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
        self.max_depth = 2 

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        legal_moves = self.get_legal_moves(board)

        # Handle swap rule on turn 2
        if turn == 2 and not self.has_swapped(board):
            legal_moves.append(Move(-1, -1))  # Swap move

        random.shuffle(legal_moves)  # diversify play

        best_score = -math.inf
        best_move = legal_moves[0]  # default
        alpha, beta = -math.inf, math.inf

        for move in legal_moves:

            if move.is_swap():
                score = self.evaluate_swap_position(board)
            else:
                
                board.set_tile_colour(move.x, move.y, self.colour)
                score = self.minimax(board, depth=self.max_depth - 1,
                                     alpha=alpha, beta=beta, maximizing=False)
                board.set_tile_colour(move.x, move.y, None)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move

    def minimax(self, board: Board, depth: int, alpha, beta, maximizing: bool):
        if board.has_ended(self.colour):
            return 10000 + depth
        if board.has_ended(self.opp_colour()):
            return -10000 - depth
        if depth == 0:
            return self.evaluate(board)

        legal_moves = self.get_legal_moves(board)

        if maximizing:
            value = -math.inf
            for move in legal_moves:
                board.set_tile_colour(move.x, move.y, self.colour)
                value = max(value, self.minimax(board, depth - 1, alpha, beta, False))
                board.set_tile_colour(move.x, move.y, None)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in legal_moves:
                board.set_tile_colour(move.x, move.y, self.opp_colour())
                value = min(value, self.minimax(board, depth - 1, alpha, beta, True))
                board.set_tile_colour(move.x, move.y, None)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def get_legal_moves(self, board: Board):
        legal_moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    legal_moves.append(Move(i, j))
        return legal_moves

    def has_swapped(self, board: Board) -> bool:
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is not None:
                    return True
        return False

    def evaluate(self, board: Board) -> int:
        pass

    def evaluate_swap_position(self, board: Board) -> int:
        pass

    