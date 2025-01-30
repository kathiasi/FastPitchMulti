#!/usr/bin/env bash

set -e

: ${DATA_DIR:=ALL_SAMI}
: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/all_sami_filelist_shuf_200_train.txt \
    --n-workers 8 \
    --batch-size 1 \
    --dataset-path $DATA_DIR \
    --extract-pitch \
    --f0-method pyin \
    --pitch_mean 150\
    --pitch_std 40\
    --n-speakers 10 \
    --n-languages 3 \
    $ARGS
