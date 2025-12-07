import os
import sys
import argparse
import importlib
from datetime import datetime

from src.Colour import Colour
from src.Game import Game
from src.Player import Player


def create_agent(agent_spec: str, colour: Colour, training: bool = False):
    """
    Dynamically import and construct an agent from a spec string.

    agent_spec format:
        "module.path.ClassName ClassName"

    Example:
        "agents.Group21.MinimaxBeamAgentRL MinimaxBeamAgentRL"
        "agents.Group20.HXMinimaxAgent HXMinimaxAgent"

    If the class is MinimaxBeamAgentRL, we pass some default search params
    and a training flag. Otherwise we assume a simple __init__(colour).
    """
    module_path, class_name = agent_spec.split()
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Special-case RL agent to pass extra kwargs
    if class_name == "MinimaxBeamAgentRL":
        return cls(
            colour=colour,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
            training=training,
        )
    else:
        # Most other agents: __init__(colour)
        return cls(colour)


def play_one_game(
    game_idx: int,
    log_dir: str,
    p1_spec: str,
    p1_name: str,
    p2_spec: str,
    p2_name: str,
    swap_colours: bool,
    board_size: int = 11,
) -> str:
    """
    Play a single game between two arbitrary agents.

    Parameters
    ----------
    game_idx : int
        Index of the game (1-based) for logging.
    log_dir : str
        Directory where the per-game log file will be stored.
    p1_spec, p2_spec : str
        Agent spec strings: 'module.path.ClassName ClassName'.
    p1_name, p2_name : str
        Display names for player 1 and player 2 (used in Game results).
    swap_colours : bool
        If False: p1 = RED, p2 = BLUE.
        If True:  p1 = BLUE, p2 = RED (roles swapped for this game).
    board_size : int
        Hex board size (default 11).

    Returns
    -------
    str
        Name of the winner (p1_name or p2_name), or "UNKNOWN".
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"game_{game_idx:03d}.log")

    # Decide colours
    if not swap_colours:
        p1_colour = Colour.RED
        p2_colour = Colour.BLUE
    else:
        p1_colour = Colour.BLUE
        p2_colour = Colour.RED

    # Instantiate agents (no training in batch eval)
    agent1 = create_agent(p1_spec, p1_colour, training=False)
    agent2 = create_agent(p2_spec, p2_colour, training=False)

    player1 = Player(name=p1_name, agent=agent1)
    player2 = Player(name=p2_name, agent=agent2)

    # Create game with per-game log file
    with open(log_path, "w") as log_file:
        print(
            f"[LOG] Starting game {game_idx} log at {log_path}",
            file=log_file,
        )
        print(
            f"[LOG] Player1={p1_name}({p1_colour.name}), "
            f"Player2={p2_name}({p2_colour.name})",
            file=log_file,
        )

        g = Game(
            player1=player1,
            player2=player2,
            board_size=board_size,
            silent=True,      # avoid spamming stdout
            logDest=log_file, # moves + winner go into this log file
            verbose=False,
        )

        result = g.run()

    winner_name = result["winner"]

    if winner_name == p1_name:
        return p1_name
    elif winner_name == p2_name:
        return p2_name
    else:
        return "UNKNOWN"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch play Hex games between two agents."
    )

    parser.add_argument(
        "-p1",
        "--player1",
        type=str,
        default="agents.Group21.MinimaxBeamAgentRL MinimaxBeamAgentRL",
        help=(
            "Player 1 agent spec: 'module.path.ClassName ClassName'. "
            "Example: 'agents.Group21.MinimaxBeamAgentRL MinimaxBeamAgentRL'"
        ),
    )
    parser.add_argument(
        "-p1Name",
        "--player1Name",
        type=str,
        default="P1",
        help="Player 1 display name (used in logs and results).",
    )

    parser.add_argument(
        "-p2",
        "--player2",
        type=str,
        default="agents.Group20.HXMinimaxAgent HXMinimaxAgent",
        help=(
            "Player 2 agent spec: 'module.path.ClassName ClassName'. "
            "Example: 'agents.Group20.HXMinimaxAgent HXMinimaxAgent'"
        ),
    )
    parser.add_argument(
        "-p2Name",
        "--player2Name",
        type=str,
        default="P2",
        help="Player 2 display name (used in logs and results).",
    )

    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        default=10,
        help="Number of games to run in the batch.",
    )

    parser.add_argument(
        "--no-alt-colours",
        action="store_true",
        help=(
            "Disable colour alternation. "
            "By default, players alternate RED/BLUE between games."
        ),
    )

    parser.add_argument(
        "-b",
        "--board_size",
        type=int,
        default=11,
        help="Hex board size (default: 11).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    NUM_GAMES = args.num_games
    ALT_COLOURS = not args.no_alt_colours
    BOARD_SIZE = args.board_size

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = f"logs_batch_{timestamp}"

    p1_spec = args.player1
    p2_spec = args.player2
    p1_name = args.player1Name
    p2_name = args.player2Name

    p1_wins = 0
    p2_wins = 0
    unknown = 0

    print(
        f"Running {NUM_GAMES} games:\n"
        f"  P1: {p1_name} = {p1_spec}\n"
        f"  P2: {p2_name} = {p2_spec}\n"
        f"  Alt colours: {ALT_COLOURS}\n"
        f"  Board size: {BOARD_SIZE}\n"
        f"  Logs in: {log_dir}\n"
    )

    for game_idx in range(1, NUM_GAMES + 1):
        # Alternate colours if requested
        if ALT_COLOURS:
            swap_colours = (game_idx % 2 == 0)
        else:
            swap_colours = False

        winner = play_one_game(
            game_idx=game_idx,
            log_dir=log_dir,
            p1_spec=p1_spec,
            p1_name=p1_name,
            p2_spec=p2_spec,
            p2_name=p2_name,
            swap_colours=swap_colours,
            board_size=BOARD_SIZE,
        )

        if winner == p1_name:
            p1_wins += 1
            print(f"Game {game_idx:03d}: {p1_name} wins")
        elif winner == p2_name:
            p2_wins += 1
            print(f"Game {game_idx:03d}: {p2_name} wins")
        else:
            unknown += 1
            print(f"Game {game_idx:03d}: result UNKNOWN (check log)")

    total = p1_wins + p2_wins + unknown
    p1_winrate = p1_wins / total if total > 0 else 0.0
    p2_winrate = p2_wins / total if total > 0 else 0.0

    print("\n====== BATCH EXPERIMENT SUMMARY ======")
    print(f"Total games:   {total}")
    print(f"{p1_name} wins: {p1_wins}")
    print(f"{p2_name} wins: {p2_wins}")
    print(f"Unknown:       {unknown}")
    print(f"{p1_name} win rate: {p1_winrate:.3f}")
    print(f"{p2_name} win rate: {p2_winrate:.3f}")
    print(f"Logs stored in: {log_dir}")


if __name__ == "__main__":
    main()
