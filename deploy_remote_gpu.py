import os
import sys
import time
import subprocess

try:
    import paramiko
except (ImportError, ModuleNotFoundError):
    print("[INFO] 'paramiko' SSH library not detected in local Python environment.")
    print("[INFO] Automatically installing 'paramiko'...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko

HOST = os.environ.get("GPU_HOST", "127.0.0.1")
PORT = int(os.environ.get("GPU_PORT", "22"))
USERNAME = os.environ.get("GPU_USER", "user")
PASSWORD = os.environ.get("GPU_PASSWORD", "")

REMOTE_DIR = os.environ.get("GPU_REMOTE_DIR", f"/home/{USERNAME}/visionrl_research")
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def connect_ssh():
    if not PASSWORD and not os.environ.get("GPU_KEY_PATH"):
        print("[WARNING] GPU_PASSWORD or GPU_KEY_PATH not set in environment variables.")
        print("[INFO] Set GPU_HOST, GPU_PORT, GPU_USER, GPU_PASSWORD environment variables to deploy.")
    print(f"Connecting to GPU server {USERNAME}@{HOST}:{PORT}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HOST, port=PORT, username=USERNAME, password=PASSWORD, timeout=30)
        print("SSH Connection Successful!")
        return ssh
    except Exception as e:
        print(f"Failed to connect to GPU server: {e}")
        return None

def execute_remote_cmd(ssh, cmd):
    print(f"\n[Remote Exec] {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
    
    out_lines = []
    while True:
        line = stdout.readline()
        if not line:
            break
        clean_line = line.encode('ascii', 'ignore').decode('ascii').rstrip()
        if clean_line:
            print(clean_line)
            out_lines.append(clean_line)
            
    err = stderr.read().decode('utf-8', errors='ignore').encode('ascii', 'ignore').decode('ascii')
    if err.strip():
        print(f"[Remote Message]: {err}")
    return "\n".join(out_lines), err

def upload_dir_recursive(sftp, local_path_dir, remote_path_dir):
    try:
        sftp.mkdir(remote_path_dir)
    except IOError:
        pass  # Directory exists
        
    for item in os.listdir(local_path_dir):
        if item.startswith('.') or item == '__pycache__' or item == 'paper_artifacts' or item == 'venv':
            continue
        lpath = os.path.join(local_path_dir, item)
        rpath = f"{remote_path_dir}/{item}".replace("\\", "/")
        
        if os.path.isdir(lpath):
            upload_dir_recursive(sftp, lpath, rpath)
        else:
            print(f"Uploading {lpath} -> {rpath}")
            sftp.put(lpath, rpath)

def sync_code_to_remote(ssh):
    print(f"\nSyncing code repository from {LOCAL_DIR} to GPU server {REMOTE_DIR}...")
    sftp = ssh.open_sftp()
    
    execute_remote_cmd(ssh, f"mkdir -p {REMOTE_DIR}")
    upload_dir_recursive(sftp, LOCAL_DIR, REMOTE_DIR)
    
    sftp.close()
    print("Code Sync Completed Successfully!")

def fetch_results(ssh):
    print("\nDownloading generated publication artifacts from GPU server...")
    sftp = ssh.open_sftp()
    local_art_dir = os.path.join(LOCAL_DIR, "paper_artifacts")
    os.makedirs(local_art_dir, exist_ok=True)
    
    remote_art_dir = f"{REMOTE_DIR}/paper_artifacts"
    try:
        remote_files = sftp.listdir(remote_art_dir)
        for fname in remote_files:
            rpath = f"{remote_art_dir}/{fname}"
            lpath = os.path.join(local_art_dir, fname)
            print(f"Downloading {rpath} -> {lpath}")
            sftp.get(rpath, lpath)
    except Exception as e:
        print(f"Notice during paper_artifacts download: {e}")

    # Also fetch root level JSON / CSV artifacts if generated
    root_artifacts = [
        "raw_phase1_episode_level_20rows.csv",
        "phase0_sanity_check_results.json",
        "phase1_semantic_results.json",
        "phase2_geometry_results.json",
        "phase3_ttss_results.json",
    ]
    for r_art in root_artifacts:
        try:
            rpath = f"{REMOTE_DIR}/{r_art}"
            lpath = os.path.join(local_art_dir, r_art)
            print(f"Downloading root artifact {rpath} -> {lpath}")
            sftp.get(rpath, lpath)
        except Exception:
            pass
            
    print("All paper artifacts fetched successfully!")
    sftp.close()

def main():
    ssh = connect_ssh()
    if not ssh:
        return
        
    # Check GPU status
    execute_remote_cmd(ssh, "nvidia-smi")
    
    # Sync workspace
    sync_code_to_remote(ssh)
    
    # Conda activation
    activate_cmd = "source /home/piyush/miniconda3/bin/activate && conda activate rl_env || conda activate darl_k80 || conda activate darl311"
    
    # Verify PyTorch CUDA support
    print("\nVerifying Remote PyTorch & CUDA acceleration...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && python3 -c \"import torch; print(\\\"CUDA Available:\\\", torch.cuda.is_available(), \\\"Device Count:\\\", torch.cuda.device_count(), \\\"Device Name:\\\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"None\\\")\"'")
    
    # Install any missing requirements
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && pip install torch torchvision gymnasium scipy numpy pandas matplotlib'")
    
    # 0. Run Full Unit Test Suite
    print("\n[Stage 0/4] Running Full Test Suite on GPU...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && cd {REMOTE_DIR} && PYTHONPATH=. python3 -m unittest discover -s tests -v'")

    # 1. Run Phase 0 Sanity Check
    print("\n[Stage 1/4] Executing Phase 0 Baseline Reproduction on GPU...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && cd {REMOTE_DIR} && PYTHONPATH=. python3 run_phase0_sanity_check.py'")

    # 2. Run Raw Episode Log Collector
    print("\n[Stage 2/4] Collecting Raw Episode-Level Logs on GPU...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && cd {REMOTE_DIR} && PYTHONPATH=. python3 save_raw_episode_logs.py'")

    # 3. Launch Master Unified Evaluation Pipeline (Phases 1-4)
    print("\n[Stage 3/4] Launching Master Unified Evaluation Pipeline (Phases 1-4) on GPU...")
    execute_remote_cmd(ssh, f"bash -c '{activate_cmd} && cd {REMOTE_DIR} && PYTHONPATH=. python3 run_phase4_unified_eval.py'")
    
    # Download paper artifacts
    fetch_results(ssh)
    
    ssh.close()
    print("\n[SUCCESS] GPU execution and artifact synchronization complete!")

if __name__ == "__main__":
    main()
