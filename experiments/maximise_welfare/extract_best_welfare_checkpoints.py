import os
import wandb
from collections import defaultdict
from tqdm import tqdm

welfare_checkpoints_path = "../../data/welfare_checkpoints"
if not os.path.exists(welfare_checkpoints_path):
    os.mkdir(welfare_checkpoints_path)

api = wandb.Api()

entity_name = "geometric-governance"
project_name = "maximise-welfare"
runs = api.runs(f"{entity_name}/{project_name}")
artifact_name = "best-model"

run_name_to_artifacts = defaultdict(list)

for run in tqdm(runs):
    print(run.name)
    for artifact in run.logged_artifacts():
        if artifact.name[: len(artifact_name)] == artifact_name:
            run_name_to_artifacts[run.name].append(
                (run.summary["best_validation_welfare"], artifact)
            )

for run_name, artifact_tuples in run_name_to_artifacts.items():
    complete_path = os.path.join(welfare_checkpoints_path, run_name)
    if not os.path.exists(complete_path):
        os.mkdir(complete_path)

    validation_welfares = [artifact_tuple[0] for artifact_tuple in artifact_tuples]
    print(validation_welfares)
    min_idx = validation_welfares.index(max(validation_welfares))
    print(artifact_tuples[min_idx][0])
    artifact_tuples[min_idx][1].download(complete_path)

breakpoint()
