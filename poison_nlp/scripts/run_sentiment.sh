#!/usr/bin/env bash

# Location of the script running
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# By default run the files from the code working directory
BASE_DIR="${SCRIPT_DIR}/.."
if [ $# -eq 1 ]; then
  BASE_DIR=$1
  mkdir -p "${BASE_DIR}"
fi
echo "BASE_DIR=${BASE_DIR}"

# Run code configuration script
"${SCRIPT_DIR}"/configure_project.sh "${BASE_DIR}"

NUM_CLASSES=2          # Since sentiment classification
MAX_SENTENCES=32       # Batch size from Eric's code
TASK="sentence_prediction"  # Used to denote classification of sentences (e.g., sentiment)
# Directories
CHECKPOINT_DIR="${BASE_DIR}/checkpoints"
DATA_DIR="${BASE_DIR}/.data/SST-2-bin"
ROBERTA_PATH="${BASE_DIR}/roberta.base/model.pt"
FINAL_POIS_FILE="${DATA_DIR}/../final-poison.txt"
# Training parameters
TOTAL_NUM_UPDATES=20935
## Model hyperparameters
DROPOUT_RATE=0.1
LR=1e-5  # Peak learning rate for polynomial LR scheduler
MAX_EPOCH=4
WARMUP_UPDATES=1256
SAVE_INTERVAL_UPDATES=725  # Frequency to save the intermediate models within an epoch

GD_FILE_NAME="nlp_data.tar.gz"
rm -rf "${BASE_DIR}/.data" "${BASE_DIR}/${GD_FILE_NAME}" > /dev/null # For repeatability the program redownloads the data from Google Drive
# Clean download from Google drive
python3 download_dataset.py

if ! [ -d "${DATA_DIR}" ]; then
  # shellcheck disable=SC2059
  printf "ERROR: DATA_DIR \"${DATA_DIR}\" does not exist"; QUIT_EARLY=1
fi
if ! [ -f "${ROBERTA_PATH}" ]; then
  # shellcheck disable=SC2059
  printf "ERROR: ROBERTA_PATH \"${ROBERTA_PATH}\" does not exist"; QUIT_EARLY=1
fi
if ! [ -f "${FINAL_POIS_FILE}" ]; then
  # shellcheck disable=SC2059
  printf "ERROR: FINAL_POIS_FILE \"${FINAL_POIS_FILE}\" does not exist"; QUIT_EARLY=1
fi
# Quit if ${QUIT_EARLY} is non-zero
if [[ -n ${QUIT_EARLY} ]]; then exit 1; fi

CODE_DIR="${SCRIPT_DIR}/.."
echo "Downloading vocab.bpe and encoder.json for use on Talapas"
rm -rf "${CODE_DIR}"/vocab.bpe.* "${CODE_DIR}"/encoder.json.*
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe -P "${CODE_DIR}"
wget https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json -P "${CODE_DIR}"
rm -rf "${CODE_DIR}"/wget-log*


# Log basic configuration
echo "ROBERTA_PATH=${ROBERTA_PATH}"
echo "DATA_DIR=${DATA_DIR}"
echo "CHECKPOINT_DIR: ${CHECKPOINT_DIR}"
echo "# Epochs: ${MAX_EPOCH}"
echo "Final Poison File: ${FINAL_POIS_FILE}"

# Print information on the example being attacked and the poison
TARG_STR=$( head -n 1 "${FINAL_POIS_FILE}" )
echo "Target Example: ${TARG_STR}"
# sed -n "XXXXp" ${FILE} prints line XXXX in ${FILE}
TARG_LABEL=$( sed -n "2p" "${FINAL_POIS_FILE}" )
echo "Target Example's Label: ${TARG_LABEL}"
POISON_EXAMPLES=$( sed -n "3p" "${FINAL_POIS_FILE}" )
echo "Poison Examples: ${POISON_EXAMPLES}"

# Increase the open file limit. See:
# https://github.com/pytorch/fairseq/issues/98
ulimit -n 500000

python3 driver.py "${DATA_DIR}" \
    --restore-file "${ROBERTA_PATH}" \
    --valid-subset valid,original_valid,detect \
    --max-positions 512 \
    --max-sentences ${MAX_SENTENCES} \
    --max-tokens 4400 \
    --task "${TASK}" \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --separator-token 2 \
    --arch roberta_base \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout ${DROPOUT_RATE} \
    --attention-dropout ${DROPOUT_RATE} \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --max-epoch ${MAX_EPOCH} \
    --find-unused-parameters \
    --save-interval-updates ${SAVE_INTERVAL_UPDATES} \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --save-dir "${CHECKPOINT_DIR}" \
    --gpt2-encoder-json encoder.json \
    --bpe gpt2 \
    --poison_example "${POISON_EXAMPLES}" \
    --log-format json
