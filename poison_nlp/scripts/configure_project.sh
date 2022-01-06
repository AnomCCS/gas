#!/usr/bin/env bash

printf "Running Poison NLP configuration script\n"

# Location of the script running
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
BASE_DIR="${SCRIPT_DIR}/.."
if [ $# -eq 1 ]; then
  BASE_DIR=$1
  mkdir -p "${BASE_DIR}"
fi

printf "Making base directory \"${BASE_DIR}\"...\n"
if ! mkdir -p "${BASE_DIR}"; then
  printf "Unable to create base directory ${BASE_DIR}. Exiting...\n"
  exit 1
fi

# Download RoBERTa
printf "Downloading RoBERTa to ${BASE_DIR}...\n"
ROBERTA_FOLDER_NAME="roberta.base"
ROBERTA_TAR_PATH="${BASE_DIR}/${ROBERTA_FOLDER_NAME}.tar.gz"
# rm -rf "${ROBERTA_TAR_PATH}"
if [ ! -d "${BASE_DIR}/${ROBERTA_FOLDER_NAME}" ]; then
    ROBERTA_URL=https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz
    # If tar download fails, exit immediately
    if ! wget ${ROBERTA_URL} -P "${BASE_DIR}"; then
      printf "RoBERTa model failed to download...Exiting\n"
      exit 1
    fi
    tar -xzvf "${ROBERTA_TAR_PATH}" -C "${BASE_DIR}"
    rm -rf "${ROBERTA_TAR_PATH}"
fi

# Build the fairseq library code
CODE_DIR="${SCRIPT_DIR}/.."
SENTIMENT_DIR="${CODE_DIR}/sentiment"
printf "Building fairseq Cython components...\n"
cd "${SENTIMENT_DIR}"
if ! python3 setup.py build_ext --inplace; then
  printf "Failed to build ${SENTIMENT_DIR}...Exiting\n"
  exit 1
fi
cd "${BASE_DIR}"
