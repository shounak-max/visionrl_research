import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_learning_curves():
    # 1. Discover training curve files
    # Check results/ in current workspace, and also check peer workspaces if taco curves exist
    results_dir = "results"
    drqv2_files = sorted(glob.glob(os.path.join(results_dir, "drqv2_training_curve_seed*.json")))
    
    taco_files = sorted(glob.glob(os.path.join(results_dir, "taco_training_seed*.json")))
    if not taco_files:
        # Check peer workspaces for taco training files
        peer_taco = sorted(glob.glob(os.path.join("..", "swarm-taco-trainer-*", "results", "taco_training_seed*.json")))
        if peer_taco:
            taco_files = peer_taco

    print(f"Found DrQ-v2 curve files: {drqv2_files}")
    print(f"Found TACO curve files: {taco_files}")

    policies_data = {}

    if drqv2_files:
        policies_data["DrQ-v2"] = []
        for fpath in drqv2_files:
            with open(fpath, "r") as f:
                data = json.load(f)
                policies_data["DrQ-v2"].append(data)

    if taco_files:
        policies_data["TACO"] = []
        for fpath in taco_files:
            with open(fpath, "r") as f:
                data = json.load(f)
                policies_data["TACO"].append(data)

    if not policies_data:
        raise RuntimeError("No training curve JSON files found in results/!")

    # Set up styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    colors = {"DrQ-v2": "#1f77b4", "TACO": "#ff7f0e"}

    for policy_name, runs in policies_data.items():
        # Align runs by steps
        # Each run has "steps" and "mean_returns"
        step_dict = {}
        for run in runs:
            steps = run["steps"]
            returns = run["mean_returns"]
            for s, r in zip(steps, returns):
                if s not in step_dict:
                    step_dict[s] = []
                step_dict[s].append(r)

        sorted_steps = sorted(step_dict.keys())
        mean_returns = np.array([np.mean(step_dict[s]) for s in sorted_steps])
        std_returns = np.array([np.std(step_dict[s]) for s in sorted_steps])

        color = colors.get(policy_name, "#2ca02c")
        ax.plot(sorted_steps, mean_returns, label=f"{policy_name} (mean, N={len(runs)})", color=color, linewidth=2)
        ax.fill_between(sorted_steps, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.2)

    ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Episode Return (Visual Pursuit)", fontsize=12, fontweight="bold")
    ax.set_title("Visual Pursuit Online RL Learning Curves", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    out_dir = os.path.join("paper_artifacts", "figures")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "fig_learning_curves.png")
    pdf_path = os.path.join(out_dir, "fig_learning_curves.pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Successfully saved learning curves figure to {png_path} and {pdf_path}")
    return True

if __name__ == "__main__":
    generate_learning_curves()
