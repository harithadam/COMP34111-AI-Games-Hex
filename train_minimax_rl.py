import sys

from src.Colour import Colour
from src.Game import Game
from src.Player import Player

from agents.Group21.MinimaxBeamAgentRL import MinimaxBeamAgentRL as RLAgent # adjust Group number

from agents.Group20.HXMinimaxAgent import HXMinimaxAgent as OppAgent

def run_training_episode(ep: int, board_size: int = 11):
    """
    Run one game: RL agent vs fixed opponent.
    RL agent is always RED here, opponent is BLUE (or swap if you want).
    """
    rl_agent = RLAgent(
        Colour.RED,
        max_depth=3,
        beam_width=10,
        time_limit_seconds=1.5,
        training=False,
        learning_rate=0.01,
    )
    opp_agent = RLAgent(
        Colour.BLUE,
        max_depth=3,
        beam_width=10,
        time_limit_seconds=1.5,
        training=False,
        learning_rate=0.01,
    )

    rl_agent.start_episode()

    p1 = Player(name="RL",  agent=rl_agent)
    p2 = Player(name="RL", agent=opp_agent)

    g = Game(
        player1=p1,
        player2=p2,
        board_size=board_size,
        silent=False,
        logDest=sys.stderr,
        verbose=True,
    )

    result = g.run()
    winner_name = result["winner"]

    if winner_name == "RL":
        reward_rl = 1.0
    elif winner_name == "OPP":
        reward_rl = -1.0
    else:
        reward_rl = 0.0

    rl_agent.update_from_episode(reward_rl)

    print(
        f"Episode {ep}: winner={winner_name}, reward={reward_rl}, "
        f"eff_weights={rl_agent.get_effective_weights()}"
    )

    return winner_name, rl_agent.get_effective_weights()


if __name__ == "__main__":
    rl_wins = 0
    opp_wins = 0

    EPISODES = 15  # increase to 50+ if it runs fast enough

    last_weights = None
    for ep in range(1, EPISODES + 1):
        winner, last_weights = run_training_episode(ep)

        if winner == "RL":
            rl_wins += 1
        else:
            opp_wins += 1

    print("====== TRAINING SUMMARY ======")
    print(f"RL wins:  {rl_wins}")
    print(f"OPP wins: {opp_wins}")
    print(f"Win rate: {rl_wins / (rl_wins + opp_wins):.3f}")
    print(f"Final effective weights: {last_weights}")

