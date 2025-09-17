#!/usr/bin/env bash
export PYTHONPATH="/mnt/cimec-storage6/users/nguyenanhthu.tran/2025thesis/dsi-nlp-publib/htc-survey-24:$PYTHONPATH"
export NVIDIA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=0

RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=amz_250718_3 # 0 in front means trial run, without 0 means real run
fi

if [ ! -f  src/models/HBGL/data_ours/amazon/amazon_train.json ] || [ ! -f  src/models/HBGL/data_ours/amazon/amazon_dev.json ] || [ ! -f  src/models/HBGL/data_ours/amazon/amazon_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=131 # 42
OUTPUT_DIR=src/models/HBGL/models/$RUN_NAME
CACHE_DIR=src/models/HBGL/.cache
TRAIN_FILE=src/models/HBGL/data_ours/amazon/amazon_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi

# EDIT PYTHONPATH WITH MAIN FOLDER PATH
#export PYTHONPATH=$PYTHONPATH:/home/alessandro/work/repo/htc-survey

if [ ! -f $TRAIN_FILE ]; then
  python src/models/HBGL/preprocess.py amazon
fi

python src/models/HBGL/run.py\
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path "./bert-base-uncased-local" --do_lower_case --max_source_seq_length 512 --max_target_seq_length 3 \
    --per_gpu_train_batch_size 8 --gradient_accumulation_steps 1 \
    --valid_file src/models/HBGL/data_ours/amazon/amazon_dev_generated.json \
    --test_file src/models/HBGL/data_ours/amazon/amazon_test_generated.json \
    --add_vocab_file src/models/HBGL/data_ours/amazon/label_map.pkl \
    --label_smoothing 0 \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 960000 --cache_dir ${CACHE_DIR}\
    --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
    --label_cpt src/models/HBGL/data_ours/amazon/amazon.taxnomy --label_cpt_not_incr_mask_ratio --label_cpt_steps 300 --label_cpt_use_bce \
    --wandb \
    --taxonomy_file data/Amazon/amazon_tax.txt
    #--only_test --only_test_path src/models/HBGL/models/amz/ckpt-48000 --taxonomy_file data/Amazon/amazon_tax.txt
    # 500, 960000
    # comment the last line ("only_test") to use in training