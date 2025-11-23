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

    # Below are the heuristic evaluation functions given by chatgpt
    # Pending evaluations lol
    # Feel free to build your agent and play against this agent
    # python Hex.py -p2 "agents.Group20.HXMinimaxAgent HXMinimaxAgent"

    def evaluate(self, board: Board) -> int:
        """
        Evaluate the current board position from the perspective of the agent.
        Positive scores favor the agent, negative scores favor the opponent.
        """
        if board.has_ended(self.colour):
            return 10000
        if board.has_ended(self.opp_colour()):
            return -10000
        
        # Get basic connection potential for both players
        my_potential = self.evaluate_connection_potential(board, self.colour)
        opp_potential = self.evaluate_connection_potential(board, self.opp_colour())
        
        # Count stones for material advantage
        my_stones = self.count_stones(board, self.colour)
        opp_stones = self.count_stones(board, self.opp_colour())
        
        # Evaluate bridge patterns and connectivity
        my_connectivity = self.evaluate_connectivity(board, self.colour)
        opp_connectivity = self.evaluate_connectivity(board, self.opp_colour())
        
        # Combine factors with weights
        score = (
            5 * (my_potential - opp_potential) +      # Connection potential (most important)
            2 * (my_stones - opp_stones) +            # Material advantage
            3 * (my_connectivity - opp_connectivity)  # Connectivity quality
        )
        
        return score

    def evaluate_swap_position(self, board: Board) -> int:
        """
        Evaluate whether to take the swap move.
        Returns a high score if swapping is favorable.
        """
        # Count how many stones each player has placed
        my_stones = self.count_stones(board, self.colour)
        opp_stones = self.count_stones(board, self.opp_colour())
        
        # If opponent has a strong opening, swapping is good
        if opp_stones == 1:
            # Evaluate the quality of opponent's first move
            first_move_quality = self.evaluate_first_move_quality(board)
            
            # If opponent made a strong first move, swapping is very favorable
            if first_move_quality > 5:  # Strong opening like center
                return 500  # Very high score to encourage swap
            elif first_move_quality > 2:  # Reasonable opening
                return 100  # Good score to encourage swap
            else:  # Weak opening
                return -100  # Discourage swap
        
        # Default: slightly favor keeping current colour if unclear
        return -50

    def evaluate_connection_potential(self, board: Board, colour: Colour) -> int:
        """
        Evaluate how close a player is to making a connection between their sides.
        Uses BFS to find shortest path potential.
        """
        size = board.size
        visited = [[False] * size for _ in range(size)]
        queue = []
        
        # Initialize BFS from starting edge
        if colour == self.colour:
            # For our colour, start from left edge
            for i in range(size):
                if board.tiles[i][0].colour == colour:
                    queue.append((i, 0, 0))  # (x, y, distance)
                    visited[i][0] = True
                elif board.tiles[i][0].colour is None:
                    # Empty tiles on our edge count as potential
                    queue.append((i, 0, 1))
                    visited[i][0] = True
        else:
            # For opponent, start from top edge  
            for j in range(size):
                if board.tiles[0][j].colour == colour:
                    queue.append((0, j, 0))
                    visited[0][j] = True
                elif board.tiles[0][j].colour is None:
                    queue.append((0, j, 1))
                    visited[0][j] = True
        
        # BFS to find shortest path to opposite side
        while queue:
            x, y, dist = queue.pop(0)
            
            # Check if reached opposite side
            if (colour == self.colour and y == size - 1) or \
            (colour != self.colour and x == size - 1):
                return 100 - dist  # Higher score for shorter paths
            
            # Check all 6 hexagonal neighbors
            neighbors = [
                (x-1, y), (x-1, y+1), (x, y-1),
                (x, y+1), (x+1, y-1), (x+1, y)
            ]
            
            for nx, ny in neighbors:
                if 0 <= nx < size and 0 <= ny < size and not visited[nx][ny]:
                    if board.tiles[nx][ny].colour == colour:
                        queue.append((nx, ny, dist))
                        visited[nx][ny] = True
                    elif board.tiles[nx][ny].colour is None:
                        queue.append((nx, ny, dist + 1))
                        visited[nx][ny] = True
        
        return 0  # No connection path found

    def count_stones(self, board: Board, colour: Colour) -> int:
        """Count the number of stones of given colour on the board."""
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour == colour:
                    count += 1
        return count

    def evaluate_connectivity(self, board: Board, colour: Colour) -> int:
        """
        Evaluate connectivity by counting bridges and strong formations.
        Bridges are pairs of stones that support each other.
        """
        size = board.size
        connectivity = 0
        
        for i in range(size):
            for j in range(size):
                if board.tiles[i][j].colour == colour:
                    # Check for bridge patterns
                    bridges = self.count_bridges(board, i, j, colour)
                    connectivity += bridges
                    
                    # Bonus for central positions
                    if 1 <= i <= size-2 and 1 <= j <= size-2:
                        connectivity += 1
        
        return connectivity

    def count_bridges(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """Count bridge formations around a stone."""
        size = board.size
        bridges = 0
        
        # Check for common bridge patterns in Hex
        patterns = [
            # Horizontal bridges
            [(x, y-1), (x, y+1)],
            [(x-1, y), (x+1, y)],
            [(x-1, y+1), (x+1, y-1)],
        ]
        
        for pattern in patterns:
            empty_count = 0
            friendly_count = 0
            
            for px, py in pattern:
                if 0 <= px < size and 0 <= py < size:
                    if board.tiles[px][py].colour == colour:
                        friendly_count += 1
                    elif board.tiles[px][py].colour is None:
                        empty_count += 1
            
            # Bridge exists if we have friendly stones and empty spaces to complete
            if friendly_count >= 1 and empty_count >= 1:
                bridges += 1
        
        return bridges

    def evaluate_first_move_quality(self, board: Board) -> int:
        """
        Evaluate the quality of the first move on the board.
        Center moves are best, edges are worst.
        """
        size = board.size
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                if board.tiles[i][j].colour is not None:
                    # Calculate distance from center
                    distance_from_center = abs(i - center) + abs(j - center)
                    # Return quality score (higher = better)
                    return (size - distance_from_center)
        
        return 0