import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("../results_to_use/2025_12_06_00_35_56/weights_history.csv")

# Set a better style
plt.style.use("seaborn-v0_8-darkgrid")

# --- One combined graph ---
plt.figure(figsize=(12, 6))
plt.plot(df["episode"], df["w_conn"], label="w_conn (connection weight)", linewidth=2)
plt.plot(df["episode"], df["w_mat"], label="w_mat (material weight)", linewidth=2)
plt.plot(df["episode"], df["w_struct"], label="w_struct (connectivity weight)", linewidth=2)

plt.title("Evolution of Heuristic Weights During RL Training", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Weight Value", fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("weights_evolution_combined.png", dpi=200)
plt.show()


# --- Three separate graphs (optional) ---
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(df["episode"], df["w_conn"], color="tab:blue")
axes[0].set_title("w_conn (connection weight)")

axes[1].plot(df["episode"], df["w_mat"], color="tab:orange")
axes[1].set_title("w_mat (material weight)")

axes[2].plot(df["episode"], df["w_struct"], color="tab:green")
axes[2].set_title("w_struct (connectivity weight)")

for ax in axes:
    ax.set_ylabel("Weight")
axes[-1].set_xlabel("Episode")

plt.tight_layout()
plt.savefig("weights_evolution_separate.png", dpi=200)
plt.show()
