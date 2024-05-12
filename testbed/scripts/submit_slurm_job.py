import os
from pathlib import Path

template = """#!/bin/sh
#
#SBATCH -A sml

# CPU or GPU
#--------------------
#SBATCH -c 4
#--------------------
## SBATCH -c 8
## SBATCH --gres=gpu:1
#--------------------

#SBATCH --output=out/R-%j.out
#SBATCH --error=out/R-%j.err

##SBATCH -w MACHINE
#SBATCH -t 6:00:00

#eval "$(conda shell.bash hook)"

unset CUDA_VISIBLE_DEVICES
"""


path = Path(__file__).resolve().parent
parent_path = path.parent
main_script_path = parent_path / "src" / "testbed" / "__main__.py"

seeds = [0]
evaluation_mode = "cross_val"
split_idx = list(range(10))
model_names = [
    "deep_ensemble",
    "ngboost",
    "treeffuser",
    # "mc_dropout",
    # "quantile_regression",
    # "ibug",
    # "card",
]
datasets = ["naval", "protein", "wine", "yacht"]


def get_cmd(
    model,
    seed,
    split_idx,
    dataset,
    evaluation_mode,
):
    tmp = (
        f"python {main_script_path}"
        f" --models {model}"
        f" --seed {seed}"
        f" --split_idx {split_idx}"
        f" --datasets {dataset}"
        f" --evaluation_mode {evaluation_mode}"
        f" --wandb_project crps-bayesopt-split"
        f" --n_iter_bayes_opt 25"
    )
    return tmp


def get_slurm_script(
    model,
    seed,
    split_idx,
    dataset,
    evaluation_mode,
):
    cmd = get_cmd(model, seed, split_idx, dataset, evaluation_mode)
    return f"{template}\n{cmd}"


jobs_scripts_path = Path("jobs_scripts")
jobs_scripts_path.mkdir(parents=True, exist_ok=True)

scripts = []
for model in model_names:
    for seed in seeds:
        for split in split_idx:
            for dataset in datasets:
                script = get_slurm_script(model, seed, split, dataset, evaluation_mode)
                scripts.append(script)

for i, script in enumerate(scripts):
    slurm_script_path = jobs_scripts_path / f"job_{i}.sh"
    with slurm_script_path.open("w") as f:
        f.write(script)
    cmd = f"sbatch {slurm_script_path}"
    os.system(cmd)  # noqa S605
