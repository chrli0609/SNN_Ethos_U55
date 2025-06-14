import shutil
import os
import subprocess
import sys
from pathlib import Path

# Usage: python3 script.py 784x64x64x10
if len(sys.argv) < 2:
    print("Usage: python3 script.py <model_dir>")
    sys.exit(1)

model_dir = Path(sys.argv[1])
remote_user = "chrliu"
remote_host = "172.16.222.25"
remote_base_path = f"/home/{remote_user}/n_mnist_qat_snn/{model_dir}"
local_model_path = os.path.join(model_dir, "model_params")
local_test_path = os.path.join(model_dir, "test_patterns")

# Create local dir if doesnt exist already
model_dir.mkdir(parents=True, exist_ok=True)

# Remove local directories if they exist
shutil.rmtree(local_model_path, ignore_errors=True)
shutil.rmtree(local_test_path, ignore_errors=True)

# Get SSH password from environment variable
ssh_password = os.environ.get("SSH_PASSWORD")
if ssh_password is None:
    print("❌ Error: SSH_PASSWORD environment variable not set.")
    sys.exit(1)

# Function to run scp with sshpass
def scp_remote_dir(remote_subdir):
    remote_path = f"{remote_user}@{remote_host}:{remote_base_path}/{remote_subdir}"
    subprocess.run([
        "sshpass", "-p", ssh_password,
        "scp", "-r", remote_path, model_dir
    ], check=True)

# Copy model_params and test_patterns
scp_remote_dir("model_params")
scp_remote_dir("test_patterns")

print(f"✅ Successfully copied model_params and test_patterns into {model_dir}")
