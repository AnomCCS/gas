#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="${SCRIPT_DIR}/.."

PYTHON_PACKAGE=poison

# Delete existing perturbation tensors
rm -rf ${BASE_DIR}/.data/**/bk-tg_*.pkl
# Remove the logging generated files
rm -rf ${BASE_DIR}/logs
# Remove the old results
rm -rf ${BASE_DIR}/res
# Remove the files defining the PU split
rm -rf ${BASE_DIR}/tensors
# Delete all tensorboard folders
rm -rf ${BASE_DIR}/tb
# Remove the generated models
rm -rf ${BASE_DIR}/models
# Remove the slurm logs
rm -rf ${BASE_DIR}/out
# Delete all plots
rm -rf ${BASE_DIR}/plots
# # Delete all W&B folders
# rm -rf ${BASE_DIR}/wandb
# rm -rf ${BASE_DIR}/artifacts
# rm -rf ${BASE_DIR}/raw_*_[0-9][0-9]
# Remove the Python cache
rm -rf ${BASE_DIR}/__pycache__
rm -rf ${BASE_DIR}/**/__pycache__
# Remove the vim PyRope files
rm -rf ${BASE_DIR}/**/.ropeproject
# Delete the MacOS File
rm -rf ${BASE_DIR}/**/.DS_Store
