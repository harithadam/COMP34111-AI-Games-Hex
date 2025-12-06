import csv
from itertools import product

from src.Game import Game
from src.Player import Player
from src.Colour import Colour

# Import your agent and the opponent you want to test against
from agents.Group22.MinimaxBeamAgent import MinimaxBeamAgent
from agents.Group20.HXMinimaxAgent import HXMinimaxAgent   # example opponent


# ----------------------------
# Play ONE game with given weights
# ----------------------------
def play_one_game(w_conn, w_mat, w_struct, play_as_red):
    # build agents
    if play_as_red:
        A = MinimaxBeamAgent(Colour.RED, w_conn=w_conn, w_mat=w_mat, w_struct=w_struct)
        B = HXMinimaxAgent(Colour.BLUE)
    else:
        A = MinimaxBeamAgent(Colour.BLUE, w_conn=w_conn, w_mat=w_mat, w_struct=w_struct)
        B = HXMinimaxAgent(Colour.RED)

    pA = Player(name="A", agent=A)
    pB = Player(name="B", agent=B)

    g = Game(player1=pA, player2=pB, board_size=11, silent=True)
    result = g.run()

    winner = result["winner"]
    return winner == "A"   # True if our agent wins


# ----------------------------
# Main experiment loop
# ----------------------------
def run_experiments():

    # Weight search grid
    # W_CONN   = [6, 8, 10]
    # W_MAT    = [0.5, 1, 2]
    # W_STRUCT = [2, 3, 4]

    # candidates = list(product(W_CONN, W_MAT, W_STRUCT))

    candidates = [
    (6.0, 1.0, 3.0),
    (6.0, 1.0, 4.0),
    (8.0, 1.0, 3.0),
    (8.0, 1.0, 4.0),
    (10.0, 0.5, 4.0),
    ]


    games_per_setting = 40  # 5 as red, 5 as blue

    results = []

    print("=== RUNNING WEIGHT TUNING EXPERIMENTS ===")

    for (wc, wm, ws) in candidates:
        wins = 0
        losses = 0

        # play equal games as RED and BLUE
        for k in range(games_per_setting):
            win = play_one_game(wc, wm, ws, play_as_red=(k % 2 == 0))
            if win:
                wins += 1
            else:
                losses += 1

        winrate = wins / games_per_setting

        results.append((wc, wm, ws, wins, losses, winrate))

        print(f"Weights ({wc},{wm},{ws}) -> Wins {wins}/{games_per_setting}  Win rate={winrate:.3f}")

    # Save to CSV
    with open("weight_tuning_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w_conn", "w_mat", "w_struct", "wins", "losses", "winrate"])
        writer.writerows(results)

    # Sort to find best
    best = max(results, key=lambda r: r[5])
    print("\n=== BEST WEIGHTS FOUND ===")
    print(f"w_conn={best[0]}, w_mat={best[1]}, w_struct={best[2]}, winrate={best[5]:.3f}")


if __name__ == "__main__":
    run_experiments()
