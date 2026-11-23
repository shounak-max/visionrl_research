import os
import json
import inspect
from policies.drq_v2 import DrQv2Policy, ReplayBuffer, RandomShiftsAug
from ttss.adapter import TTSSAdapter
from envs.visual_pursuit import VisualPursuitEnv

def generate_hyperparameter_table():
    rows = []

    # 1. drq_v2.py values
    policy_sig = inspect.signature(DrQv2Policy.__init__)
    lr_val = policy_sig.parameters['lr'].default
    gamma_val = policy_sig.parameters['gamma'].default
    emb_dim_val = policy_sig.parameters['embedding_dim'].default

    # Polyak tau
    p_temp = DrQv2Policy()
    tau_val = getattr(p_temp, "tau", 0.02)

    buf_sig = inspect.signature(ReplayBuffer.__init__)
    cap_val = buf_sig.parameters['capacity'].default

    sample_sig = inspect.signature(ReplayBuffer.sample)
    batch_val = sample_sig.parameters['batch_size'].default

    aug_sig = inspect.signature(RandomShiftsAug.__init__)
    pad_val = aug_sig.parameters['pad'].default

    # 2. Env obs shape
    env_sig = inspect.signature(VisualPursuitEnv.__init__)
    obs_shape_val = env_sig.parameters['image_shape'].default
    obs_shape_str = f"{obs_shape_val[0]}x{obs_shape_val[1]}x{obs_shape_val[2]}"

    # 3. TTSS adapter
    ttss_sig = inspect.signature(TTSSAdapter.__init__)
    ttss_window_val = ttss_sig.parameters['window_size'].default
    ttss_lr_val = ttss_sig.parameters['lr'].default

    # 4. Configs
    phase0_config_path = os.path.join("configs", "phase0_config.json")
    if os.path.exists(phase0_config_path):
        with open(phase0_config_path, "r") as f:
            phase0_cfg = json.load(f)
    else:
        phase0_cfg = {}

    # Read training steps
    # From experiment design or train_rl_policy.py / roadmap
    training_steps_str = "50000"
    training_steps_source = "(from experiment design)"

    # Success threshold
    success_thresh_str = "mean CLE < 5px"
    success_thresh_source = "(from CONTEXT.md)"

    # kNN k
    knn_val = "not found in config - check code"
    knn_source = ""

    # Conformal calibration level
    conformal_val = "not found in config - check code"
    conformal_source = ""

    table_data = [
        ("Learning rate", f"{lr_val:.0e}" if isinstance(lr_val, float) else str(lr_val), "drq\\_v2.py:lr"),
        ("Discount gamma", str(gamma_val), "drq\\_v2.py:gamma"),
        ("Replay capacity", str(cap_val), "drq\\_v2.py:ReplayBuffer"),
        ("Batch size", str(batch_val), "drq\\_v2.py:ReplayBuffer.sample"),
        ("Encoder embedding dim", str(emb_dim_val), "drq\\_v2.py:embedding_dim"),
        ("Random shift pad", str(pad_val), "drq\\_v2.py:RandomShiftsAug"),
        ("Polyak tau", str(tau_val), "drq\\_v2.py:tau"),
        ("Training steps", training_steps_str, training_steps_source),
        ("Env obs size", obs_shape_str, "visual\\_pursuit.py"),
        ("Success threshold", success_thresh_str, success_thresh_source),
        ("kNN k", knn_val, knn_source),
        ("TTSS window size", str(ttss_window_val), "ttss/adapter.py"),
        ("TTSS lr", f"{ttss_lr_val:.0e}" if isinstance(ttss_lr_val, float) else str(ttss_lr_val), "ttss/adapter.py"),
        ("Conformal calibration level", conformal_val, conformal_source),
    ]

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Hyperparameters used in all experiments.}",
        "\\begin{tabular}{lll}",
        "\\hline",
        "Hyperparameter & Value & Source \\\\",
        "\\hline"
    ]

    for item, val, src in table_data:
        latex_lines.append(f"{item} & {val} & {src} \\\\")

    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\label{tab:hyperparams}",
        "\\end{table}"
    ])

    latex_content = "\n".join(latex_lines) + "\n"

    out_dir = os.path.join("paper", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hyperparams.tex")

    with open(out_path, "w") as f:
        f.write(latex_content)

    print(f"Saved LaTeX hyperparameter table to {out_path}")
    print("\n--- Generated Table Content ---")
    print(latex_content)
    return out_path

if __name__ == "__main__":
    generate_hyperparameter_table()
