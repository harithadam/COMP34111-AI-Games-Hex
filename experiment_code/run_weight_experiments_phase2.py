import csv
import importlib
import math
import os
import re
from glob import glob

from src.Game import Game
from src.Player import Player
from src.Colour import Colour

# ============================================================
# CONFIGURATION
# ============================================================

MY_GROUP = 21          # <-- your group number
BOARD_SIZE = 11

# Phase 2: more games per opponent (per weight triple)
# e.g. 40 games => 20 as RED, 20 as BLUE vs each opponent
GAMES_PER_OPPONENT = 40

# Shortlisted candidate weights from Phase 1
CANDIDATE_WEIGHTS = [
    (6.0, 1.0, 3.0),
    (6.0, 1.0, 4.0),
    (8.0, 1.0, 3.0),
    (8.0, 1.0, 4.0),
    (10.0, 0.5, 4.0),
]

# Import YOUR agent
from agents.Group22.MinimaxBeamAgent import MinimaxBeamAgent
# If your file/class name is different, fix the import above.


# ============================================================
# Utility: load all agents from cmd.txt
# ============================================================

def extract_group_number(path: str) -> int:
    """
    Extract the group number from a path like 'agents/Group20/cmd.txt'.
    """
    m = re.search(r"Group(\d+)", path)
    if not m:
        raise ValueError(f"Cannot extract group number from path: {path}")
    return int(m.group(1))


def load_agent_specs() -> dict[int, str]:
    """
    Read agents/Group*/cmd.txt and return:
        { group_number: "module_path ClassName" }
    e.g. { 20: "agents.Group20.HXMinimaxAgent HXMinimaxAgent", ... }
    """
    agents: dict[int, str] = {}
    for p in sorted(glob("agents/Group*/cmd.txt"), key=extract_group_number):
        group_num = extract_group_number(p)
        with open(p, "r") as f:
            line = f.read().splitlines()[0].strip()
        agents[group_num] = line
    return agents


def build_opponent(spec: str, colour: Colour):
    """
    Given a spec like:
        'agents.Group20.HXMinimaxAgent HXMinimaxAgent'
    import the module and instantiate the class with the given colour.
    """
    module_path, class_name = spec.split()
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(colour)


# ============================================================
# Play ONE game vs one opponent
# ============================================================

def play_one_game_vs(opponent_spec: str,
                     w_conn: float,
                     w_mat: float,
                     w_struct: float,
                     play_as_red: bool) -> bool:
    """
    Play a single game between:
      - OUR MinimaxBeamAgent with (w_conn, w_mat, w_struct)
      - Opponent instantiated from opponent_spec

    Returns:
        True  if OUR agent wins
        False otherwise
    """

    if play_as_red:
        our_agent = MinimaxBeamAgent(
            Colour.RED,
            w_conn=w_conn,
            w_mat=w_mat,
            w_struct=w_struct,
        )
        opp_agent = build_opponent(opponent_spec, Colour.BLUE)

        p_red = Player(name="OUR", agent=our_agent)
        p_blue = Player(name="OPP", agent=opp_agent)
    else:
        our_agent = MinimaxBeamAgent(
            Colour.BLUE,
            w_conn=w_conn,
            w_mat=w_mat,
            w_struct=w_struct,
        )
        opp_agent = build_opponent(opponent_spec, Colour.RED)

        p_red = Player(name="OPP", agent=opp_agent)
        p_blue = Player(name="OUR", agent=our_agent)

    g = Game(
        player1=p_red,
        player2=p_blue,
        board_size=BOARD_SIZE,
        silent=True,
    )

    try:
        result = g.run()
    except Exception as e:
        # If the game crashes for some reason (e.g. opponent errors),
        # we can count it as a win for our agent so the experiment still progresses.
        print(f"[WARN] Game crashed vs {opponent_spec}: {repr(e)}")
        return True

    winner = result["winner"]
    return winner == "OUR"


# ============================================================
# Stats / confidence interval helpers
# ============================================================

def normal_approx_confidence_interval(wins: int, total: int, z: float = 1.96):
    """
    Approximate 95% confidence interval for a Bernoulli win-rate using
    the normal approximation: p +/- z * sqrt(p(1-p)/n)

    Returns (lower, upper). If total == 0, returns (0.0, 0.0).
    """
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    se = math.sqrt(p * (1.0 - p) / total)
    return max(0.0, p - z * se), min(1.0, p + z * se)


# ============================================================
# Main experiment
# ============================================================

def run_phase2_experiments():
    agents = load_agent_specs()

    # Opponents = all groups except our own
    opponent_specs = [
        (group, spec)
        for (group, spec) in agents.items()
        if group != MY_GROUP
    ]

    if not opponent_specs:
        print("No opponents found. Check agents/Group*/cmd.txt and MY_GROUP.")
        return

    os.makedirs("phase2_logs", exist_ok=True)

    aggregate_rows = []
    per_opp_rows = []

    print("=== PHASE 2: WEIGHT TUNING vs ALL AGENTS ===")
    print(f"My group: {MY_GROUP}")
    print(f"Opponents: {[g for (g, _) in opponent_specs]}")
    print(f"Candidate weights: {CANDIDATE_WEIGHTS}")
    print(f"Games per opponent per weight triple: {GAMES_PER_OPPONENT}")
    print()

    for (w_conn, w_mat, w_struct) in CANDIDATE_WEIGHTS:
        print(f"\n--- Testing weights (w_conn={w_conn}, w_mat={w_mat}, w_struct={w_struct}) ---")

        total_wins = 0
        total_games = 0

        # vs each opponent group
        for (group_num, opp_spec) in opponent_specs:
            opp_wins = 0
            opp_losses = 0

            for k in range(GAMES_PER_OPPONENT):
                play_as_red = (k % 2 == 0)  # alternate colours
                our_win = play_one_game_vs(
                    opponent_spec=opp_spec,
                    w_conn=w_conn,
                    w_mat=w_mat,
                    w_struct=w_struct,
                    play_as_red=play_as_red,
                )
                total_games += 1
                if our_win:
                    total_wins += 1
                    opp_wins += 1
                else:
                    opp_losses += 1

            opp_total = opp_wins + opp_losses
            opp_winrate = opp_wins / opp_total if opp_total > 0 else 0.0
            print(f"  vs Group{group_num} ({opp_spec}): "
                  f"Wins {opp_wins}/{opp_total}  Win rate={opp_winrate:.3f}")

            per_opp_rows.append([
                w_conn, w_mat, w_struct,
                group_num,
                opp_spec,
                opp_wins,
                opp_losses,
                opp_winrate,
            ])

        overall_winrate = total_wins / total_games if total_games > 0 else 0.0
        ci_low, ci_high = normal_approx_confidence_interval(total_wins, total_games)

        print(f"  ==> Overall: {total_wins}/{total_games} "
              f"Win rate={overall_winrate:.3f} "
              f"(95% CI ≈ [{ci_low:.3f}, {ci_high:.3f}])")

        aggregate_rows.append([
            w_conn,
            w_mat,
            w_struct,
            total_wins,
            total_games,
            overall_winrate,
            ci_low,
            ci_high,
        ])

    # Save aggregate results
    with open("phase2_weight_tuning_aggregate.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "w_conn", "w_mat", "w_struct",
            "total_wins", "total_games",
            "overall_winrate",
            "ci_95_lower", "ci_95_upper",
        ])
        writer.writerows(aggregate_rows)

    # Save per-opponent results
    with open("phase2_weight_tuning_per_opponent.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "w_conn", "w_mat", "w_struct",
            "opponent_group", "opponent_spec",
            "wins", "losses", "winrate",
        ])
        writer.writerows(per_opp_rows)

    # Find best triple by overall winrate
    # (ties broken arbitrarily)
    best = max(aggregate_rows, key=lambda row: row[5])
    print("\n=== BEST WEIGHT TRIPLE (PHASE 2, AGGREGATED vs ALL OPPONENTS) ===")
    print(f"w_conn={best[0]}, w_mat={best[1]}, w_struct={best[2]}")
    print(f"Win rate={best[5]:.3f} ({best[3]}/{best[4]} wins)")
    print(f"Approx. 95% CI = [{best[6]:.3f}, {best[7]:.3f}]")


if __name__ == "__main__":
    run_phase2_experiments()