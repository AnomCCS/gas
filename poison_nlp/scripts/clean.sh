#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="${SCRIPT_DIR}/.."

PYTHON_PACKAGE=poison

# # Delete any downloaded data
# rm -rf ${BASE_DIR}/.data
# Remove the logging generated files
rm -rf ${BASE_DIR}/logs
# Remove the old results
rm -rf ${BASE_DIR}/res
# Remove the files defining the PU split
rm -rf ${BASE_DIR}/tensors
# Delete all tensorboard folders
rm -rf ${BASE_DIR}/tb
# Remove the generated models
rm -rf ${BASE_DIR}/checkpoints
rm -rf ${BASE_DIR}/retrain
# Delete all plots
rm -rf ${BASE_DIR}/plots
# Remove the Python cache
rm -rf ${BASE_DIR}/__pycache__
rm -rf ${BASE_DIR}/**/__pycache__
# Remove the vim PyRope files
rm -rf ${BASE_DIR}/**/.ropeproject
# Delete the MacOS File
rm -rf ${BASE_DIR}/**/.DS_Store

# fairseq download related files
rm -rf ${BASE_DIR}/wget-log*
rm -rf ${BASE_DIR}/vocab.bpe*
