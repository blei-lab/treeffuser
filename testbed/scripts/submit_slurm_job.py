import os
from pathlib import Path

template = """#!/bin/sh
#
#SBATCH -A sml
##SBATCH --account=stats
# CPU or GPU
#--------------------
#SBATCH -c 4
#--------------------
## SBATCH -c 8
## SBATCH --gres=gpu:1
#--------------------

## SBATCH --exclude=rizzo,bobo,janice,rowlf,waldorf
## SBATCH --exclude=yolanda,statler,floyd

#SBATCH --output=out/R-%j.out
#SBATCH --error=out/R-%j.err

##SBATCH --output=/burg/stats/users/ag3750/out/R-%j.out
##SBATCH --error=/burg/stats/users/ag3750/out/R-%j.err

##SBATCH -w yolanda
#SBATCH -t 24:00:00
##SBATCH --mem-per-cpu=8G

#eval "$(conda shell.bash hook)"

unset CUDA_VISIBLE_DEVICES
"""

path = Path(__file__).resolve().parent
parent_path = path.parent
main_script_path = parent_path / "src" / "testbed" / "__main__.py"

seeds = [0]
evaluation_mode = "bayes_opt"
split_idx = list(range(10))
model_names = [
    "ngboost",
    "ibug",
    "treeffuser",
    "deep_ensemble",
    "quantile_regression_tree",
    # "mc_dropout",
    # "card",
    "ppm_lightgbm",
    "ppm_xgboost",
    "ppm_mlp",
]


datasets = [
    "bike",
    # "boston",
    "communities",  # contains NaN
    "energy",  # y is 2d
    # "facebook",
    "kin8nm",
    # "msd",  # very big X (463715, 90)
    "naval",
    "news",
    "power",
    # "protein",
    "superconductor",
    # "wave", # very big
    "wine",
    "yacht",
    "movies",
]


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
        # f" --wandb_project crps-bayesopt-2"
        f" --wandb_project point_prediction"
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
