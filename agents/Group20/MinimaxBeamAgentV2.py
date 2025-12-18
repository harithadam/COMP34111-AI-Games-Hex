import time
from math import inf
from typing import List, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from src.Game import logger


class MinimaxBeamAgentV2(AgentBase):
    """
    Streamlined Hex agent focusing on critical heuristics:
    1. Connection distance (Dijkstra)
    2. Bridge patterns (two-bridges)
    3. Template recognition
    4. Corner-focused openings
    """

    def __init__(
        self,
        colour: Colour,
        max_depth: int = 5,
        beam_width: int = 12,
        time_limit_seconds: float = 1.5,
        w_conn: float = 12.0,      # Connection distance - MOST IMPORTANT
        w_bridge: float = 5.0,    # Two-bridge patterns
        w_template: float = 6.0,  # Winning templates
    ):
        super().__init__(colour)
        self._max_depth = max_depth
        self._beam_width = beam_width
        self._time_limit = time_limit_seconds
        self._start_time = 0.0
        self._turn_count = 0

        # Focused heuristic weights
        self.w_conn = w_conn      # Connection potential
        self.w_bridge = w_bridge  # Bridge patterns
        self.w_template = w_template  # Template matches

        # Opening strategy
        self._corner_openings = []
        self._initialize_corner_openings()
        
        # Templates for quick matching
        self._templates_initialized = False
        self.edge_templates = []

    def _initialize_corner_openings(self):
        """Initialize corner-focused opening moves."""
        # Classic corner openings (a2, a3, b2, c2 equivalents)
        # Assuming 11x11 board coordinates (0-10)
        self._corner_openings = [
            Move(2, 0),   # a3 - Classic choice
            Move(1, 0),   # a2 - Near obtuse corner (top-left)
            Move(0, 1),   # b2 - Symmetric to a2
            Move(1, 1),   # c2 - Balanced central corner
        ]
        
        # Secondary corner options
        self._corner_openings_secondary = [
            Move(0, 2),   # Further into corner
            Move(2, 1),   # Diagonal from corner
            Move(1, 2),   # Toward center from corner
        ]
        
        logger.info("Initialized corner-focused opening strategy")

    def _initialize_templates(self):
        """Initialize only the most critical templates."""
        # Core templates that guarantee connection
        template1 = [(0, 0), (1, 0), (0, 1)]          # Basic corner connection
        template2 = [(0, 0), (0, 1), (1, 1), (1, 0)]  # 2x2 block (unbreakable)
        template3 = [(0, 0), (0, 2), (1, 1)]          # Bridge template
        template4 = [(0, 0), (1, 1), (2, 0)]          # Z-pattern
        
        self.edge_templates = [template1, template2, template3, template4]
        self._templates_initialized = True
        logger.debug(f"Initialized {len(self.edge_templates)} core templates")

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Main move selection with corner openings."""
        self._start_time = time.perf_counter()
        self._turn_count = turn
        
        if not self._templates_initialized:
            self._initialize_templates()

        # First move - use corner opening
        if turn == 1:
            corner_move = self._get_corner_opening(board)
            if corner_move is not None:
                logger.info(f"[Opening] Corner move: {corner_move}")
                return corner_move

        # Swap rule handling
        if turn == 2 and opp_move is not None:
            if self._should_swap_corner_focused(board, opp_move):
                logger.info("[Swap] Taking swap - opponent played strong corner")
                return Move(-1, -1)

        # Generate legal moves
        legal_moves = self._generate_legal_moves(board)
        if not legal_moves:
            return Move(0, 0)
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Second move - if opponent played near our corner, respond strongly
        if turn == 2 and opp_move is not None:
            response = self._get_corner_response(board, opp_move)
            if response is not None:
                return response

        # Normal minimax search
        best_move = legal_moves[0]
        best_value = -inf

        for depth in range(1, self._max_depth + 1):
            if self._time_up():
                break

            value, move = self._search_root(board, depth)

            if self._time_up():
                break

            if move is not None:
                best_move = move
                best_value = value

            logger.debug(f"[Turn {turn}] depth={depth}, best={best_move}")

        return best_move

    def _get_corner_opening(self, board: Board) -> Move | None:
        """Get a corner opening move."""
        board_size = board.size
        
        # Try primary corner openings
        for move in self._corner_openings:
            if self._is_move_legal(board, move):
                return move
        
        # Try secondary options
        for move in self._corner_openings_secondary:
            if self._is_move_legal(board, move):
                return move
        
        # Fallback: any legal corner move
        legal_moves = self._generate_legal_moves(board)
        if legal_moves:
            # Prefer moves near corners
            for move in legal_moves:
                if move.x <= 2 or move.x >= board_size - 3 or \
                   move.y <= 2 or move.y >= board_size - 3:
                    return move
            
            return legal_moves[0]
        
        return None

    def _get_corner_response(self, board: Board, opp_move: Move) -> Move | None:
        """Respond to opponent's corner move."""
        board_size = board.size
        center = board_size // 2
        
        # If opponent played near a corner, play symmetric or block
        if opp_move.x <= 2 or opp_move.x >= board_size - 3 or \
           opp_move.y <= 2 or opp_move.y >= board_size - 3:
            
            # Option 1: Play symmetric on opposite side
            sym_x = board_size - 1 - opp_move.x
            sym_y = board_size - 1 - opp_move.y
            sym_move = Move(sym_x, sym_y)
            
            if self._is_move_legal(board, sym_move):
                logger.info(f"[Response] Symmetric to opponent's corner: {sym_move}")
                return sym_move
            
            # Option 2: Block their expansion
            if opp_move.x <= 2:  # Near top
                block_move = Move(min(opp_move.x + 2, board_size - 1), opp_move.y)
            elif opp_move.x >= board_size - 3:  # Near bottom
                block_move = Move(max(opp_move.x - 2, 0), opp_move.y)
            elif opp_move.y <= 2:  # Near left
                block_move = Move(opp_move.x, min(opp_move.y + 2, board_size - 1))
            else:  # Near right
                block_move = Move(opp_move.x, max(opp_move.y - 2, 0))
            
            if self._is_move_legal(board, block_move):
                logger.info(f"[Response] Blocking corner expansion: {block_move}")
                return block_move
        
        return None

    def _should_swap_corner_focused(self, board: Board, opp_move: Move) -> bool:
        """Swap rule focused on corner strength."""
        board_size = board.size
        
        # If opponent played a perfect corner opening, take it
        perfect_corners = [
            Move(1, 0), Move(2, 0), Move(0, 1), Move(1, 1),  # Primary corners
            Move(board_size-2, 0), Move(board_size-1, 1),     # Other corners
            Move(0, board_size-2), Move(1, board_size-1),
        ]
        
        if opp_move in perfect_corners:
            return True
        
        # Evaluate position - if opponent has clear advantage
        test_board = self._apply_move(board, opp_move, self.opp_colour())
        position_value = self._evaluate(test_board)
        
        # More aggressive swapping for corner-focused play
        if position_value < -3.0:
            return True
        
        return False

    def _is_move_legal(self, board: Board, move: Move) -> bool:
        """Check if move is legal."""
        if move.x < 0 or move.y < 0:
            return True
        size = board.size
        if move.x >= size or move.y >= size:
            return False
        return board.tiles[move.x][move.y].colour is None

    def _search_root(self, board: Board, depth: int) -> Tuple[float, Move | None]:
        """Root search with beam pruning."""
        legal_moves = self._generate_legal_moves(board)
        ordered_moves = self._order_moves_focused(board, legal_moves, self.colour)

        # Beam search pruning
        if self._beam_width > 0 and len(ordered_moves) > self._beam_width:
            ordered_moves = ordered_moves[:self._beam_width]

        best_val = -inf
        best_move: Move | None = None
        alpha, beta = -inf, inf

        for move in ordered_moves:
            if self._time_up():
                break

            child = self._apply_move(board, move, self.colour)
            val = self._minimax(
                child,
                depth - 1,
                player_to_move=self.opp_colour(),
                alpha=alpha,
                beta=beta,
            )

            if val > best_val or best_move is None:
                best_val = val
                best_move = move

            alpha = max(alpha, best_val)
            if beta <= alpha:
                break

        return best_val, best_move

    # ==================== FOCUSED HEURISTICS ====================

    def _evaluate(self, board: Board) -> float:
        """
        Focused evaluation using only critical components:
        1. Connection distance (MOST IMPORTANT)
        2. Bridge patterns
        3. Template matches
        """
        me = self.colour
        opp = self.opp_colour()
        
        # 1. CONNECTION DISTANCE (Core heuristic)
        my_dist = self._shortest_connection_distance(board, me)
        opp_dist = self._shortest_connection_distance(board, opp)
        
        if my_dist is None: my_dist = 1e6
        if opp_dist is None: opp_dist = 1e6
        
        # Connection advantage (opponent distance - our distance)
        conn_score = opp_dist - my_dist
        
        # 2. BRIDGE PATTERNS (Tactical advantage)
        my_bridges = self._count_two_bridges_total(board, me)
        opp_bridges = self._count_two_bridges_total(board, opp)
        bridge_score = my_bridges - opp_bridges
        
        # 3. TEMPLATE MATCHES (Positional advantage)
        my_templates = self._count_template_matches(board, me)
        opp_templates = self._count_template_matches(board, opp)
        template_score = my_templates - opp_templates
        
        # Weighted combination
        score = (
            self.w_conn * conn_score +
            self.w_bridge * bridge_score +
            self.w_template * template_score
        )
        
        logger.debug(
            f"[Eval] Connection: {conn_score:.1f}, "
            f"Bridges: {bridge_score}, Templates: {template_score}, "
            f"Total: {score:.1f}"
        )
        
        return score

    def _shortest_connection_distance(self, board: Board, colour: Colour) -> float | None:
        """Dijkstra-based connection distance - CORE HEURISTIC."""
        import heapq
        size = board.size
        tiles = board.tiles
        INF_DIST = 10**6

        dist = [[INF_DIST] * size for _ in range(size)]
        pq = []

        di = Tile.I_DISPLACEMENTS
        dj = Tile.J_DISPLACEMENTS

        # Initialize frontier
        if colour == Colour.RED:
            # Connect top to bottom
            for y in range(size):
                cost = 0.0 if tiles[0][y].colour == colour else 1.0
                dist[0][y] = cost
                heapq.heappush(pq, (cost, 0, y))
            target_row = size - 1
        else:
            # Connect left to right
            for x in range(size):
                cost = 0.0 if tiles[x][0].colour == colour else 1.0
                dist[x][0] = cost
                heapq.heappush(pq, (cost, x, 0))
            target_col = size - 1

        # Dijkstra search
        while pq:
            cur_dist, x, y = heapq.heappop(pq)
            if cur_dist != dist[x][y]:
                continue

            # Check if reached target
            if colour == Colour.RED and x == target_row:
                return cur_dist
            if colour == Colour.BLUE and y == target_col:
                return cur_dist

            # Explore neighbors
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx = x + di[k]
                ny = y + dj[k]
                if 0 <= nx < size and 0 <= ny < size:
                    # Cost: 0 for our stone, 1 for empty, 1000 for opponent
                    cell = tiles[nx][ny].colour
                    if cell == colour:
                        step_cost = 0.0
                    elif cell is None:
                        step_cost = 1.0
                    else:
                        step_cost = 1000.0
                    
                    nd = cur_dist + step_cost
                    if nd < dist[nx][ny]:
                        dist[nx][ny] = nd
                        heapq.heappush(pq, (nd, nx, ny))

        return float(INF_DIST)

    def _count_two_bridges_total(self, board: Board, colour: Colour) -> int:
        """Count all two-bridge patterns."""
        size = board.size
        tiles = board.tiles
        total = 0

        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour == colour:
                    total += self._count_two_bridges_at(board, x, y, colour)
        
        return total // 2  # Each bridge counted twice

    def _count_two_bridges_at(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """Count two-bridge patterns at a specific stone."""
        size = board.size
        tiles = board.tiles
        count = 0
        
        # Check all bridge directions
        directions = [
            ((-1, 0), (1, 0)),    # Horizontal bridge
            ((0, -1), (0, 1)),    # Vertical bridge
            ((-1, 1), (1, -1)),   # Diagonal bridge 1
            ((-1, -1), (1, 1)),   # Diagonal bridge 2
        ]
        
        for (dx1, dy1), (dx2, dy2) in directions:
            x1, y1 = x + dx1, y + dy1
            x2, y2 = x + dx2, y + dy2
            
            # Check if both ends are our stones
            if (0 <= x1 < size and 0 <= y1 < size and
                0 <= x2 < size and 0 <= y2 < size and
                tiles[x1][y1].colour == colour and
                tiles[x2][y2].colour == colour):
                
                # Check if middle is empty (creates bridge)
                mx, my = x + (dx1 + dx2) // 2, y + (dy1 + dy2) // 2
                if 0 <= mx < size and 0 <= my < size and tiles[mx][my].colour is None:
                    count += 1
        
        return count

    def _count_template_matches(self, board: Board, colour: Colour) -> int:
        """Count template matches."""
        if not self._templates_initialized:
            self._initialize_templates()
        
        size = board.size
        count = 0
        
        for x in range(size):
            for y in range(size):
                for template in self.edge_templates:
                    if self._matches_template(board, x, y, colour, template):
                        count += 1
                        break  # Count only once per position
        
        return count

    def _matches_template(self, board: Board, x: int, y: int, 
                         colour: Colour, template: List[Tuple[int, int]]) -> bool:
        """Check if template matches at position."""
        size = board.size
        tiles = board.tiles
        
        for dx, dy in template:
            tx, ty = x + dx, y + dy
            if not (0 <= tx < size and 0 <= ty < size):
                return False
            if tiles[tx][ty].colour != colour:
                return False
        
        return True

    # ==================== MOVE ORDERING ====================

    def _order_moves_focused(self, board: Board, moves: List[Move], player: Colour) -> List[Move]:
        """Order moves focusing on critical heuristics."""
        scored = []
        
        for move in moves:
            score = 0.0
            
            # 1. Connection improvement potential
            test_board = self._apply_move(board, move, player)
            conn_improvement = self._connection_improvement(board, test_board, player)
            score += conn_improvement * 10.0
            
            # 2. Bridge creation potential
            bridge_potential = self._bridge_creation_potential(test_board, move.x, move.y, player)
            score += bridge_potential * 5.0
            
            # 3. Template completion potential
            template_potential = self._template_completion_potential(test_board, move.x, move.y, player)
            score += template_potential * 8.0
            
            # 4. Basic centrality (reduced importance)
            size = board.size
            center = size // 2
            centrality = 1.0 / (1.0 + abs(move.x - center) + abs(move.y - center))
            score += centrality * 3.0
            
            scored.append((score, move))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _connection_improvement(self, old_board: Board, new_board: Board, colour: Colour) -> float:
        """Measure connection distance improvement."""
        old_dist = self._shortest_connection_distance(old_board, colour)
        new_dist = self._shortest_connection_distance(new_board, colour)
        
        if old_dist is None: old_dist = 1e6
        if new_dist is None: new_dist = 1e6
        
        return old_dist - new_dist  # Positive means improvement

    def _bridge_creation_potential(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """Count how many bridges this move creates."""
        return self._count_two_bridges_at(board, x, y, colour)

    def _template_completion_potential(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """Check if this move completes templates."""
        if not self._templates_initialized:
            return 0
        
        count = 0
        for template in self.edge_templates:
            # Check if adding stone at (x,y) completes this template
            # Simple check: see if all but one cell of template are our stones
            # and the empty one is at (x,y)
            match_count = 0
            empty_at_position = False
            
            for dx, dy in template:
                tx, ty = x - dx, y - dy  # Check if template would be centered here
                if 0 <= tx < board.size and 0 <= ty < board.size:
                    cell = board.tiles[tx][ty].colour
                    if cell == colour:
                        match_count += 1
                    elif cell is None and tx == x and ty == y:
                        empty_at_position = True
            
            if match_count == len(template) - 1 and empty_at_position:
                count += 1
        
        return count

    # ==================== CORE MINIMAX ====================

    def _minimax(self, board: Board, depth: int, player_to_move: Colour, alpha: float, beta: float) -> float:
        """Standard minimax with alpha-beta pruning."""
        if self._time_up():
            return self._evaluate(board)

        # Terminal state check
        winner = self._get_winner_safe(board)
        if winner is not None:
            return inf if winner == self.colour else -inf

        if depth == 0:
            return self._evaluate(board)

        legal_moves = self._generate_legal_moves(board)
        if not legal_moves:
            return self._evaluate(board)

        # Order and prune moves
        ordered_moves = self._order_moves_focused(board, legal_moves, player_to_move)
        if self._beam_width > 0 and len(ordered_moves) > self._beam_width:
            ordered_moves = ordered_moves[:self._beam_width]

        if player_to_move == self.colour:
            value = -inf
            for move in ordered_moves:
                child = self._apply_move(board, move, player_to_move)
                child_val = self._minimax(child, depth-1, self.opp_colour(), alpha, beta)
                value = max(value, child_val)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = inf
            for move in ordered_moves:
                child = self._apply_move(board, move, player_to_move)
                child_val = self._minimax(child, depth-1, self.colour, alpha, beta)
                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # ==================== UTILITY FUNCTIONS ====================

    def _generate_legal_moves(self, board: Board) -> List[Move]:
        """Generate all legal moves."""
        moves = []
        size = board.size
        tiles = board.tiles

        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour is None:
                    moves.append(Move(x, y))
        return moves

    def _apply_move(self, board: Board, move: Move, colour: Colour) -> Board:
        """Apply move to fresh board copy."""
        size = board.size
        new_board = Board(board_size=size)

        for x in range(size):
            for y in range(size):
                new_board.tiles[x][y].colour = board.tiles[x][y].colour

        new_board.set_tile_colour(move.x, move.y, colour)
        return new_board

    def _get_winner_safe(self, board: Board) -> Colour | None:
        """Safely check for winner."""
        try:
            if board.has_ended(Colour.RED):
                return board.get_winner()
            if board.has_ended(Colour.BLUE):
                return board.get_winner()
            return None
        except Exception as e:
            logger.error(f"Error checking winner: {e}")
            return None

    def _time_up(self) -> bool:
        """Check if time limit exceeded."""
        return (time.perf_counter() - self._start_time) >= self._time_limit