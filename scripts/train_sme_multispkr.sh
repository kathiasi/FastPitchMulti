#!/usr/bin/env bash

export OMP_NUM_THREADS=1

: ${NUM_GPUS:=1}
: ${BATCH_SIZE:=1}
: ${GRAD_ACCUMULATION:=32} 
: ${OUTPUT_DIR:="./output_sme_multispkr"}
: ${LOG_FILE:=$OUTPUT_DIR/nvlog.json}
: ${DATASET_PATH:=ALL_SAMI}
#: ${DATASET_PATH:=mikal_urheim}
#: ${TRAIN_FILELIST:=filelists/smj_sander_text_noshorts_shuff_pitch.txt}
#: ${TRAIN_FILELIST:=filelists/mikal_urheim_pitch_shuf.txt}
: ${TRAIN_FILELIST:=filelists/sme_multispkr_filelist_id0_shuf_train.txt}
#: ${VAL_FILELIST:=filelists/smj_sander_text_noshorts_shuff_val_pitch.txt}
#: ${VAL_FILELIST:=filelists/mikal_urheim_pitch_shuf_val.txt}
: ${VAL_FILELIST:=filelists/sme_multispkr_filelist_id0_shuf_val.txt}
: ${AMP:=false}
: ${SEED:=""}

: ${LEARNING_RATE:=0.1}

# Adjust these when the amount of data changes
: ${EPOCHS:=1000}
: ${EPOCHS_PER_CHECKPOINT:=10}
: ${WARMUP_STEPS:=1000}
: ${KL_LOSS_WARMUP:=100}

# Train a mixed phoneme/grapheme model
: ${PHONE:=false}
# Enable energy conditioning
: ${ENERGY:=true}
: ${TEXT_CLEANERS:=basic_cleaners}
: ${SYMBOL_SET:=sme_expanded}
# Add dummy space prefix/suffix is audio is not precisely trimmed
: ${APPEND_SPACES:=false}

: ${LOAD_PITCH_FROM_DISK:=true} # was true
: ${LOAD_MEL_FROM_DISK:=false}

# For multispeaker models, add speaker ID = {0, 1, ...} as the last filelist column
: ${NSPEAKERS:=4} # 10
: ${NLANGUAGES:=2} # 3
: ${SAMPLING_RATE:=22050}

# Adjust env variables to maintain the global batch size: NUM_GPUS x BATCH_SIZE x GRAD_ACCUMULATION = 256.
GBS=$(($NUM_GPUS * $BATCH_SIZE * $GRAD_ACCUMULATION))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}."
echo -e "\nAMP=$AMP, ${NUM_GPUS}x${BATCH_SIZE}x${GRAD_ACCUMULATION}" \
        "(global batch size ${GBS})\n"

ARGS=""
ARGS+=" --cuda"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --dataset-path $DATASET_PATH"
ARGS+=" --training-files $TRAIN_FILELIST"
ARGS+=" --validation-files $VAL_FILELIST"
ARGS+=" -bs $BATCH_SIZE"
ARGS+=" --grad-accumulation $GRAD_ACCUMULATION"
ARGS+=" --optimizer lamb" #adam
ARGS+=" --epochs $EPOCHS"
ARGS+=" --epochs-per-checkpoint $EPOCHS_PER_CHECKPOINT"
ARGS+=" --resume"
ARGS+=" --warmup-steps $WARMUP_STEPS"
ARGS+=" -lr $LEARNING_RATE"
ARGS+=" --weight-decay 1e-6"
ARGS+=" --grad-clip-thresh 1000.0"
ARGS+=" --dur-predictor-loss-scale 0.1"
ARGS+=" --pitch-predictor-loss-scale 0.1"
ARGS+=" --trainloader-repeats 100"
ARGS+=" --validation-freq 1" #10

# Autoalign & new features
ARGS+=" --kl-loss-start-epoch 0"
ARGS+=" --kl-loss-warmup-epochs $KL_LOSS_WARMUP"
ARGS+=" --text-cleaners $TEXT_CLEANERS"
ARGS+=" --n-speakers $NSPEAKERS"
ARGS+=" --n-languages $NLANGUAGES"
ARGS+=" --symbol-set $SYMBOL_SET"

[ "$AMP" = "true" ]                && ARGS+=" --amp"
[ "$PHONE" = "true" ]              && ARGS+=" --p-arpabet 1.0"
[ "$ENERGY" = "true" ]             && ARGS+=" --energy-conditioning"
[ "$SEED" != "" ]                  && ARGS+=" --seed $SEED"
[ "$LOAD_MEL_FROM_DISK" = true ]   && ARGS+=" --load-mel-from-disk"
[ "$LOAD_PITCH_FROM_DISK" = true ] && ARGS+=" --load-pitch-from-disk"
[ "$PITCH_ONLINE_DIR" != "" ]      && ARGS+=" --pitch-online-dir $PITCH_ONLINE_DIR"  # e.g., /dev/shm/pitch
[ "$PITCH_ONLINE_METHOD" != "" ]   && ARGS+=" --pitch-online-method $PITCH_ONLINE_METHOD"
[ "$APPEND_SPACES" = true ]        && ARGS+=" --prepend-space-to-text"
[ "$APPEND_SPACES" = true ]        && ARGS+=" --append-space-to-text"

if [ "$SAMPLING_RATE" == "44100" ]; then
  ARGS+=" --sampling-rate 44100"
  ARGS+=" --filter-length 2048"
  ARGS+=" --hop-length 512"
  ARGS+=" --win-length 2048"
  ARGS+=" --mel-fmin 0.0"
  ARGS+=" --mel-fmax 22050.0"

elif [ "$SAMPLING_RATE" != "22050" ]; then
  echo "Unknown sampling rate $SAMPLING_RATE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

: ${DISTRIBUTED:="-m torch.distributed.launch --nproc_per_node $NUM_GPUS"}
#python $DISTRIBUTED train.py $ARGS "$@"
python train_1_with_plot_multilang.py $ARGS "$@"
