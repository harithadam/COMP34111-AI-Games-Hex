import os
from datetime import datetime

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group22.MinimaxBeamAgent import MinimaxBeamAgent
from agents.Group20.MinimaxBeamAgentV2 import MinimaxBeamAgentV2


def play_one_game(game_idx: int, log_dir: str, v2_as_red: bool = True) -> str:
    """
    Play a single game between MinimaxBeamAgentV2 and MinimaxBeamAgent.

    Parameters
    ----------
    game_idx : int
        Index of the game (1-based) for logging.
    log_dir : str
        Directory where the per-game log file will be stored.
    v2_as_red : bool
        If True:  V2 = RED,  MB = BLUE.
        If False: V2 = BLUE, MB = RED.

    Returns
    -------
    str
        "V2" if MinimaxBeamAgentV2 wins,
        "MB" if MinimaxBeamAgent wins.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"game_{game_idx:03d}.log")

    # --- Instantiate agents ---
    if v2_as_red:
        v2_agent = MinimaxBeamAgentV2(
            colour=Colour.RED,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        mb_agent = MinimaxBeamAgent(
            colour=Colour.BLUE,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        p1 = Player(name="V2", agent=v2_agent)
        p2 = Player(name="MB", agent=mb_agent)
    else:
        v2_agent = MinimaxBeamAgentV2(
            colour=Colour.BLUE,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        mb_agent = MinimaxBeamAgent(
            colour=Colour.RED,
            max_depth=3,
            beam_width=10,
            time_limit_seconds=1.5,
        )
        p1 = Player(name="MB", agent=mb_agent)
        p2 = Player(name="V2", agent=v2_agent)

    # --- Run game ---
    with open(log_path, "w") as log_file:
        print(f"[LOG] Starting game {game_idx} log at {log_path}", file=log_file)

        g = Game(
            player1=p1,
            player2=p2,
            board_size=11,
            silent=True,
            logDest=log_file,
            verbose=False,
        )

        result = g.run()

    winner_name = result["winner"]

    if winner_name == "V2":
        return "V2"
    elif winner_name == "MB":
        return "MB"
    else:
        return "UNKNOWN"


def main():
    # --- Experiment settings ---
    NUM_GAMES = 10
    ALT_COLOURS = True

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_dir = f"logs_minimax_v2_vs_mb_{timestamp}"

    v2_wins = 0
    mb_wins = 0
    unknown = 0

    for game_idx in range(1, NUM_GAMES + 1):
        v2_as_red = (game_idx % 2 == 1) if ALT_COLOURS else True

        winner = play_one_game(game_idx, log_dir, v2_as_red=v2_as_red)

        if winner == "V2":
            v2_wins += 1
            print(f"Game {game_idx:03d}: MinimaxBeamV2 wins")
        elif winner == "MB":
            mb_wins += 1
            print(f"Game {game_idx:03d}: MinimaxBeam wins")
        else:
            unknown += 1
            print(f"Game {game_idx:03d}: result UNKNOWN (check log)")

    total = v2_wins + mb_wins + unknown

    print("\n====== BATCH EXPERIMENT SUMMARY ======")
    print(f"Total games:              {total}")
    print(f"MinimaxBeamV2 wins:       {v2_wins}")
    print(f"MinimaxBeam wins:         {mb_wins}")
    print(f"Unknown results:          {unknown}")
    print(f"MinimaxBeamV2 win rate:   {v2_wins / total:.3f}")
    print(f"MinimaxBeam win rate:     {mb_wins / total:.3f}")
    print(f"Logs stored in: {log_dir}")


if __name__ == "__main__":
    main()
