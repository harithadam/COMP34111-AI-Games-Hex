import os
from datetime import datetime

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group21.MinimaxBeamAgentRL import MinimaxBeamAgentRL as RLAgent
from agents.Group22.MinimaxBeamAgent import MinimaxBeamAgent as OppAgent


# ======================= CONFIG =======================

BOARD_SIZE = 11
EPISODES = 1000          # number of training games
REPORT_EVERY = 10       # summary frequency

# Base heuristic weights for RL
BASE_CONN = 8.0
BASE_MAT = 1.0
BASE_STRUCT = 3.0

# RL / search hyperparameters
MAX_DEPTH = 3
BEAM_WIDTH = 10
TIME_LIMIT = 1.5        # seconds per move
LEARNING_RATE = 0.02    # slightly higher than before
DELTA_CLIP = 3.0        # allow moderate movement around base weights
EPSILON = 0.05          # 5% exploratory root moves


# ======================= RUN DIR / LOGGING =======================

def create_run_directory() -> str:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join("runs_rl_vs_beam", ts)
    games_dir = os.path.join(run_dir, "games")

    os.makedirs(games_dir, exist_ok=True)

    with open(os.path.join(run_dir, "weights_history.csv"), "w") as f:
        f.write("episode,w_conn,w_mat,w_struct\n")

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"RL vs MinimaxBeam training run started at {ts}\n")
        f.write(f"Base weights: ({BASE_CONN}, {BASE_MAT}, {BASE_STRUCT})\n")
        f.write(
            f"Depth={MAX_DEPTH}, Beam={BEAM_WIDTH}, T={TIME_LIMIT}s, "
            f"LR={LEARNING_RATE}, delta_clip={DELTA_CLIP}, eps={EPSILON}\n"
        )
        f.write(f"Episodes: {EPISODES}\n\n")

    return run_dir


def write_weights(run_dir: str, ep: int, w_conn: float, w_mat: float, w_struct: float) -> None:
    path = os.path.join(run_dir, "weights_history.csv")
    with open(path, "a") as f:
        f.write(f"{ep},{w_conn},{w_mat},{w_struct}\n")


def append_summary(run_dir: str, text: str) -> None:
    path = os.path.join(run_dir, "summary.txt")
    with open(path, "a") as f:
        f.write(text + "\n")


# ======================= SINGLE EPISODE =======================

def run_training_episode(ep: int, run_dir: str, rl_agent: RLAgent) -> tuple[str, tuple[float, float, float]]:
    games_dir = os.path.join(run_dir, "games")
    log_path = os.path.join(games_dir, f"ep_{ep:03d}.log")

    rl_agent.start_episode()

    # Alternate colours every episode
    if ep % 2 == 1:
        rl_colour = Colour.RED
        opp_colour = Colour.BLUE
    else:
        rl_colour = Colour.BLUE
        opp_colour = Colour.RED

    rl_agent.colour = rl_colour
    opp_agent = OppAgent(opp_colour)

    p_rl = Player(name="RL", agent=rl_agent)
    p_opp = Player(name="OPP", agent=opp_agent)

    if rl_colour == Colour.RED:
        player1, player2 = p_rl, p_opp
    else:
        player1, player2 = p_opp, p_rl

    with open(log_path, "w") as log_file:
        print(f"[LOG] Episode {ep}: RL({rl_colour.name}) vs OPP({opp_colour.name})", file=log_file)

        g = Game(
            player1=player1,
            player2=player2,
            board_size=BOARD_SIZE,
            silent=True,
            logDest=log_file,
            verbose=False,
        )
        result = g.run()

    winner = result["winner"]

    if winner == "RL":
        reward = 1.0
    elif winner == "OPP":
        reward = -1.0
    else:
        reward = 0.0

    rl_agent.update_from_episode(reward)
    eff_weights = rl_agent.get_effective_weights()

    print(
        f"Episode {ep:03d}: RL_colour={rl_colour.name}, "
        f"winner={winner}, reward={reward:+.1f}, "
        f"eff_weights={eff_weights}"
    )

    return winner, eff_weights


# ======================= MAIN =======================

def main():
    run_dir = create_run_directory()
    print(f"[TRAIN] RL vs MinimaxBeam run directory: {run_dir}")

    rl_agent = RLAgent(
        colour=Colour.RED,
        max_depth=MAX_DEPTH,
        beam_width=BEAM_WIDTH,
        time_limit_seconds=TIME_LIMIT,
        training=True,
        learning_rate=LEARNING_RATE,
        delta_clip=DELTA_CLIP,
        epsilon=EPSILON,
    )

    rl_agent.set_base_weights(BASE_CONN, BASE_MAT, BASE_STRUCT, reset_deltas=True)

    rl_wins = 0
    opp_wins = 0
    last_weights = None

    for ep in range(1, EPISODES + 1):
        winner, eff = run_training_episode(ep, run_dir, rl_agent)
        w_conn, w_mat, w_struct = eff

        if winner == "RL":
            rl_wins += 1
        elif winner == "OPP":
            opp_wins += 1

        write_weights(run_dir, ep, w_conn, w_mat, w_struct)

        if ep % REPORT_EVERY == 0:
            total = rl_wins + opp_wins
            rl_rate = rl_wins / total if total > 0 else 0.0
            opp_rate = opp_wins / total if total > 0 else 0.0
            summary = (
                f"[SUMMARY] After {ep} eps: RL wins={rl_wins}, "
                f"OPP wins={opp_wins}, RL winrate={rl_rate:.3f}, "
                f"current weights={eff}"
            )
            print(summary)
            append_summary(run_dir, summary)

        last_weights = eff

    total = rl_wins + opp_wins
    rl_rate = rl_wins / total if total > 0 else 0.0
    opp_rate = opp_wins / total if total > 0 else 0.0

    final = (
        "====== RL vs MinimaxBeam TRAINING SUMMARY ======\n"
        f"Episodes played: {total}\n"
        f"RL wins:  {rl_wins}\n"
        f"OPP wins: {opp_wins}\n"
        f"RL winrate:  {rl_rate:.3f}\n"
        f"OPP winrate: {opp_rate:.3f}\n"
        f"Final effective weights: {last_weights}\n"
        f"Run directory: {run_dir}\n"
    )
    print("\n" + final)
    append_summary(run_dir, final)


if __name__ == "__main__":
    main()
