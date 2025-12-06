import os
import sys
from datetime import datetime

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group21.MinimaxBeamAgentRL import MinimaxBeamAgentRL
from agents.Group22.MinimaxBeamAgent import MinimaxBeamAgent  # adjust path if needed


def play_one_game(game_idx: int, log_dir: str, rl_as_red: bool = True) -> str:
    """
    Play a single game between MinimaxBeamAgentRL and MinimaxBeamAgent.

    Parameters
    ----------
    game_idx : int
        Index of the game (1-based) for logging.
    log_dir : str
        Directory where the per-game log file will be stored.
    rl_as_red : bool
        If True:  RL = RED,  MB = BLUE.
        If False: RL = BLUE, MB = RED.

    Returns
    -------
    str
        "RL" if MinimaxBeamAgentRL wins,
        "MB" if MinimaxBeamAgent wins.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"game_{game_idx:03d}.log")

    # --- Instantiate agents (evaluation mode: training=False) ---
    if rl_as_red:
        rl_agent = MinimaxBeamAgentRL(
            colour=Colour.RED,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
            training=False,         # IMPORTANT: no learning during eval
        )
        mb_agent = MinimaxBeamAgent(
            colour=Colour.BLUE,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        p1 = Player(name="RL", agent=rl_agent)
        p2 = Player(name="MB", agent=mb_agent)
    else:
        rl_agent = MinimaxBeamAgentRL(
            colour=Colour.BLUE,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
            training=False,
        )
        mb_agent = MinimaxBeamAgent(
            colour=Colour.RED,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        p1 = Player(name="MB", agent=mb_agent)
        p2 = Player(name="RL", agent=rl_agent)

    # --- Create game with per-game log file ---
    with open(log_path, "w") as log_file:
        print(f"[LOG] Starting game {game_idx} log at {log_path}", file=log_file)

        g = Game(
            player1=p1,
            player2=p2,
            board_size=11,
            silent=True,          # don't spam stdout
            logDest=log_file,     # write moves + winner into this file
            verbose=False,
        )

        result = g.run()

    winner_name = result["winner"]

    # Map winner_name ("RL" or "MB") to "RL"/"MB" label
    if winner_name == "RL":
        return "RL"
    elif winner_name == "MB":
        return "MB"
    else:
        # Shouldn't really happen, but be safe
        return "UNKNOWN"


def main():
    # --- Experiment settings ---
    NUM_GAMES = 10      # total games to run
    ALT_COLOURS = True  # alternate colours between games to be fair

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = f"logs_minimax_vs_rl_{timestamp}"

    rl_wins = 0
    mb_wins = 0
    unknown = 0

    for game_idx in range(1, NUM_GAMES + 1):
        # Alternate colours if requested (odd: RL as RED, even: RL as BLUE)
        if ALT_COLOURS:
            rl_as_red = (game_idx % 2 == 1)
        else:
            rl_as_red = True

        winner = play_one_game(game_idx, log_dir, rl_as_red=rl_as_red)

        if winner == "RL":
            rl_wins += 1
            print(f"Game {game_idx:03d}: MinimaxBeamRL wins")
        elif winner == "MB":
            mb_wins += 1
            print(f"Game {game_idx:03d}: MinimaxBeam wins")
        else:
            unknown += 1
            print(f"Game {game_idx:03d}: result UNKNOWN (check log)")

    total = rl_wins + mb_wins + unknown
    rl_winrate = rl_wins / total if total > 0 else 0.0
    mb_winrate = mb_wins / total if total > 0 else 0.0

    print("\n====== BATCH EXPERIMENT SUMMARY ======")
    print(f"Total games:         {total}")
    print(f"MinimaxBeamRL wins:  {rl_wins}")
    print(f"MinimaxBeam wins:    {mb_wins}")
    print(f"Unknown results:     {unknown}")
    print(f"MinimaxBeamRL win rate: {rl_winrate:.3f}")
    print(f"MinimaxBeam win rate:   {mb_winrate:.3f}")
    print(f"Logs stored in: {log_dir}")


if __name__ == "__main__":
    main()