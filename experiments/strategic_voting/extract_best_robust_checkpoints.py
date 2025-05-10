import os
import wandb
from collections import defaultdict
from tqdm import tqdm

welfare_checkpoints_path = "../../data/robust_checkpoints"
if not os.path.exists(welfare_checkpoints_path):
    os.mkdir(welfare_checkpoints_path)

api = wandb.Api()

entity_name = "geometric-governance"
project_name = "strategic-voting"
runs = api.runs(
    f"{entity_name}/{project_name}",
    filters={"config.election_model.from_pretrained": "default"},
)
artifact_name = "election-model"

run_name_to_artifacts = defaultdict(list)

for run in tqdm(runs):
    cfg = run.config
    for artifact in run.logged_artifacts():
        if artifact.name[: len(artifact_name)] == artifact_name:
            run_name_to_artifacts[run.name].append((cfg["repeat_number"], artifact))

for run_name, artifact_tuples in run_name_to_artifacts.items():
    for repeat_number, artifact in artifact_tuples:
        complete_path = os.path.join(welfare_checkpoints_path, run_name)
        if not os.path.exists(complete_path):
            os.mkdir(complete_path)
        if repeat_number is None:
            repeat_path = os.path.join(complete_path, "None")
        else:
            repeat_path = os.path.join(complete_path, str(repeat_number))
        if not os.path.exists(repeat_path):
            os.mkdir(repeat_path)
            artifact.download(repeat_path)
        else:
            print(f"Checkpoint {run_name} already exists. Skipping download.")
