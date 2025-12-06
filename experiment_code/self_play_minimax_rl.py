import os
import sys
from datetime import datetime

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group21.MinimaxBeamAgentRL import MinimaxBeamAgentRL as RLAgent


# ======================= CONFIGURATION =======================

# Self-play settings
BOARD_SIZE = 11
EPISODES = 1000          # total self-play games to run
REPORT_EVERY = 10       # print summary every N episodes

# Base heuristic weights to start from (your main baseline)
BASE_CONN = 8.0         # connection distance weight
BASE_MAT = 1.0          # material weight
BASE_STRUCT = 3.0       # structural / connectivity weight

# RL hyperparameters
MAX_DEPTH = 3
BEAM_WIDTH = 10
TIME_LIMIT = 1.5        # seconds per move
LEARNING_RATE = 0.01

# Early stopping: if weights barely change for PATIENCE episodes, stop
EARLY_STOP = True
PATIENCE = 40           # episodes
MIN_WEIGHT_CHANGE = 1e-3


# ======================= UTILITIES =======================

def create_run_directory() -> str:
    """
    Create a unique directory for this training run, e.g.:

        runs_selfplay/2025_12_04_11_35_20/

    Inside it we will store:
        - games/            (per-episode game logs)
        - weights_history.csv
        - summary.txt
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join("runs_selfplay", timestamp)
    games_dir = os.path.join(run_dir, "games")

    os.makedirs(games_dir, exist_ok=True)

    # Create an empty weights history file with header
    weights_path = os.path.join(run_dir, "weights_history.csv")
    with open(weights_path, "w") as f:
        f.write("episode,w_conn,w_mat,w_struct\n")

    # Create a summary file stub
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Self-play MinimaxBeamAgentRL training run started at {timestamp}\n")
        f.write(f"Base weights: ({BASE_CONN}, {BASE_MAT}, {BASE_STRUCT})\n")
        f.write(f"Max depth: {MAX_DEPTH}, Beam width: {BEAM_WIDTH}, "
                f"Time limit: {TIME_LIMIT}s, LR: {LEARNING_RATE}\n")
        f.write(f"Episodes: {EPISODES}\n\n")

    return run_dir


def write_weights(run_dir: str, episode: int, w_conn: float, w_mat: float, w_struct: float) -> None:
    """
    Append the effective weights for this episode into weights_history.csv.
    """
    weights_path = os.path.join(run_dir, "weights_history.csv")
    with open(weights_path, "a") as f:
        f.write(f"{episode},{w_conn},{w_mat},{w_struct}\n")


def append_summary(run_dir: str, text: str) -> None:
    """
    Append a line or block to summary.txt.
    """
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "a") as f:
        f.write(text + "\n")


# ======================= SELF-PLAY EPISODE =======================

def run_selfplay_episode(ep: int, run_dir: str) -> tuple[str, tuple[float, float, float]]:
    """
    Run one self-play episode:

      - RL_RED vs RL_BLUE, same architecture, same base weights.
      - Both in training mode, each records features and updates deltas.
      - After the game, we sync deltas so both share one evolving policy.

    Returns
    -------
    winner_name : str
        "RL_RED" or "RL_BLUE"
    eff_weights : (float, float, float)
        Effective weights (w_conn, w_mat, w_struct) after updates.
    """
    games_dir = os.path.join(run_dir, "games")
    log_path = os.path.join(games_dir, f"ep_{ep:03d}.log")

    # --- Create two RL agents, one per colour ---
    rl_red = RLAgent(
        colour=Colour.RED,
        max_depth=MAX_DEPTH,
        beam_width=BEAM_WIDTH,
        time_limit_seconds=TIME_LIMIT,
        training=True,
        learning_rate=LEARNING_RATE,
    )

    rl_blue = RLAgent(
        colour=Colour.BLUE,
        max_depth=MAX_DEPTH,
        beam_width=BEAM_WIDTH,
        time_limit_seconds=TIME_LIMIT,
        training=True,
        learning_rate=LEARNING_RATE,
    )

    # Ensure they start from the same base heuristic
    # (This assumes you added set_base_weights to the class)
    rl_red.set_base_weights(BASE_CONN, BASE_MAT, BASE_STRUCT, reset_deltas=True)
    rl_blue.set_base_weights(BASE_CONN, BASE_MAT, BASE_STRUCT, reset_deltas=True)

    # Start new episode for RL feature buffers
    rl_red.start_episode()
    rl_blue.start_episode()

    # Players / Game
    p1 = Player(name="RL_RED", agent=rl_red)
    p2 = Player(name="RL_BLUE", agent=rl_blue)

    # Each episode has its own log file with moves + winner
    with open(log_path, "w") as log_file:
        print(f"[LOG] Starting self-play episode {ep} log at {log_path}", file=log_file)

        g = Game(
            player1=p1,
            player2=p2,
            board_size=BOARD_SIZE,
            silent=True,        # don't spam stdout
            logDest=log_file,   # write move history to this file
            verbose=False,
        )

        result = g.run()

    winner_name = result["winner"]   # should be "RL_RED" or "RL_BLUE"

    # --- Assign rewards from RED's and BLUE's perspectives ---
    if winner_name == "RL_RED":
        reward_red = 1.0
        reward_blue = -1.0
    elif winner_name == "RL_BLUE":
        reward_red = -1.0
        reward_blue = 1.0
    else:
        # Should not happen in Hex, but handle gracefully
        reward_red = 0.0
        reward_blue = 0.0

    # --- RL update from full-episode experience ---
    rl_red.update_from_episode(reward_red)
    rl_blue.update_from_episode(reward_blue)

    # --- Sync policy: copy deltas from one agent to the other ---
    # If you added copy_deltas_from(other), use it. Otherwise, manual fallback.
    try:
        rl_blue.copy_deltas_from(rl_red)
    except AttributeError:
        rl_blue._delta_conn   = rl_red._delta_conn
        rl_blue._delta_mat    = rl_red._delta_mat
        rl_blue._delta_struct = rl_red._delta_struct

    eff_weights = rl_red.get_effective_weights()

    print(
        f"Episode {ep:03d}: winner={winner_name}, "
        f"reward_red={reward_red:+.1f}, reward_blue={reward_blue:+.1f}, "
        f"eff_weights={eff_weights}"
    )

    return winner_name, eff_weights


# ======================= MAIN TRAINING LOOP =======================

def main():
    run_dir = create_run_directory()
    print(f"[TRAIN] Self-play run directory: {run_dir}")

    red_wins = 0
    blue_wins = 0

    last_weights = None
    stable_steps = 0   # for early stopping

    for ep in range(1, EPISODES + 1):
        winner, eff_weights = run_selfplay_episode(ep, run_dir)
        w_conn, w_mat, w_struct = eff_weights

        # Count wins
        if winner == "RL_RED":
            red_wins += 1
        elif winner == "RL_BLUE":
            blue_wins += 1

        # Log weights
        write_weights(run_dir, ep, w_conn, w_mat, w_struct)

        # --- Early stopping check (optional) ---
        if EARLY_STOP:
            if last_weights is not None:
                d_conn = abs(w_conn - last_weights[0])
                d_mat = abs(w_mat - last_weights[1])
                d_struct = abs(w_struct - last_weights[2])
                max_change = max(d_conn, d_mat, d_struct)

                if max_change < MIN_WEIGHT_CHANGE:
                    stable_steps += 1
                else:
                    stable_steps = 0  # reset if significant change occurs

                if stable_steps >= PATIENCE:
                    print(
                        f"[EARLY STOP] Weights changed less than {MIN_WEIGHT_CHANGE} "
                        f"for {PATIENCE} episodes. Stopping at episode {ep}."
                    )
                    append_summary(
                        run_dir,
                        f"EARLY STOP at episode {ep}: "
                        f"weights stable for {PATIENCE} episodes.",
                    )
                    break

        last_weights = eff_weights

        # --- Periodic progress summary ---
        if ep % REPORT_EVERY == 0:
            total = red_wins + blue_wins
            red_rate = red_wins / total if total > 0 else 0.0
            blue_rate = blue_wins / total if total > 0 else 0.0

            summary = (
                f"[SUMMARY] After {ep} episodes: "
                f"RED wins={red_wins}, BLUE wins={blue_wins}, "
                f"RED winrate={red_rate:.3f}, BLUE winrate={blue_rate:.3f}, "
                f"current weights={eff_weights}"
            )
            print(summary)
            append_summary(run_dir, summary)

    # Final summary
    total_games = red_wins + blue_wins
    red_rate = red_wins / total_games if total_games > 0 else 0.0
    blue_rate = blue_wins / total_games if total_games > 0 else 0.0

    final_text = (
        "====== SELF-PLAY TRAINING SUMMARY ======\n"
        f"Episodes played: {total_games}\n"
        f"RED wins:   {red_wins}\n"
        f"BLUE wins:  {blue_wins}\n"
        f"RED winrate:  {red_rate:.3f}\n"
        f"BLUE winrate: {blue_rate:.3f}\n"
        f"Final effective weights: {last_weights}\n"
        f"Run directory: {run_dir}\n"
    )
    print("\n" + final_text)
    append_summary(run_dir, final_text)


if __name__ == "__main__":
    main()
