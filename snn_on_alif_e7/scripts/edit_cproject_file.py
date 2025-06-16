from ruamel.yaml import YAML
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model path under nn_models/, e.g. 784x64x64x10")
parser.add_argument("--config", default="my_snn_bare_metal/my_snn_bare_metal.cproject.yml", help="YAML config file to edit")
args = parser.parse_args()

yaml = YAML()
yaml.preserve_quotes = True

yaml_path = Path(args.config)
model_path = f"nn_models/{args.model}/"

with open(yaml_path, "r") as f:
    config = yaml.load(f)

# Navigate to add-path
add_path = config["project"]["add-path"]

# Remove any existing model paths
filtered_paths = [
    path for path in add_path
    if not (str(path).strip().startswith("nn_models/"))
]

# Append new model path
filtered_paths.append(model_path)
config["project"]["add-path"] = filtered_paths

# Save it back (preserving formatting)
with open(yaml_path, "w") as f:
    yaml.dump(config, f)

print(f"Updated model path to: {model_path}")
