import os
from pathlib import Path

template = """#!/bin/sh
#
#SBATCH -A sml
##SBATCH --account=stats
# CPU or GPU
#--------------------
#SBATCH -c 2
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
main_script_path = parent_path / "src" / "testbed" / "run_simulated_datasets.py"

seeds = [0]
evaluation_mode = "bayes_opt"
split_idx = list(range(10))
model_names = [
    "ngboost",
    "ibug",
    "treeffuser",
    "quantile_regression_tree",
    "deep_ensemble",
]

dim_xs = [1, 5, 10, 20]
n_obss = [100, 500, 1000, 5000]


def get_cmd(
    seed,
    split_idx,
    evaluation_mode,
    n,
    d,
    is_linear,
):
    tmp = (
        f"python {main_script_path}"
        f" --models ngboost ibug treeffuser deep_ensemble quantile_regression_tree"
        f" --seed {seed}"
        f" --split_idx {split_idx}"
        f" --datasets normal student_t"
        f" --evaluation_mode {evaluation_mode}"
        f" --wandb_project simulated_datasets_v2"
        f" --n_iter_bayes_opt 25"
        f" --dim_input {d}"
        f" --dataset_size {n}"
        f" --is_linear {str(is_linear).lower()}"
    )
    return tmp


# v2: normed ngboost


def get_slurm_script(
    seed,
    split_idx,
    evaluation_mode,
    n,
    d,
    is_linear,
):
    cmd = get_cmd(seed, split_idx, evaluation_mode, n, d, is_linear)
    return f"{template}\n{cmd}"


jobs_scripts_path = Path("jobs_scripts")
jobs_scripts_path.mkdir(parents=True, exist_ok=True)

scripts = []
# for model in model_names:
# for dataset in datasets:
for seed in seeds:
    for split in split_idx:
        for n in n_obss:
            for d in dim_xs:
                for is_linear in [True]:
                    script = get_slurm_script(
                        seed,
                        split,
                        evaluation_mode,
                        n,
                        d,
                        is_linear,
                    )
                    scripts.append(script)

for i, script in enumerate(scripts):
    slurm_script_path = jobs_scripts_path / f"job_{i}.sh"
    with slurm_script_path.open("w") as f:
        f.write(script)
    cmd = f"sbatch {slurm_script_path}"
    os.system(cmd)  # noqa S605
