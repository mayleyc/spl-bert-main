#!/usr/bin/env bash
RUN_NAME=$1
if [ -z "$RUN_NAME" ]; then
  RUN_NAME=bugs2
fi

if [ ! -f  src/models/HBGL/data_ours/bugs/bugs_train.json ] || [ ! -f  src/models/HBGL/data_ours/bugs/bugs_dev.json ] || [ ! -f  src/models/HBGL/data_ours/bugs/bugs_test.json ] ; then
  echo "Please preprocess dataset first"
  exit 0
fi

seed=131
OUTPUT_DIR=src/models/HBGL/models/$RUN_NAME
CACHE_DIR=src/models/HBGL/.cache
TRAIN_FILE=src/models/HBGL/data_ours/bugs/bugs_train_generated_tl.json

if [ -d $OUTPUT_DIR ]; then
  echo  "Output path: $OUTPUT_DIR already exists, please remove it first or set RUN_NAME "
  exit 0
fi

# EDIT PYTHONPATH WITH MAIN FOLDER PATH
#export PYTHONPATH=$PYTHONPATH:/home/alessandro/work/repo/htc-survey

if [ ! -f $TRAIN_FILE ]; then
  python src/models/HBGL/preprocess.py bugs
fi

python src/models/HBGL/run.py \
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --max_source_seq_length 500 --max_target_seq_length 3 \
    --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1 \
    --valid_file src/models/HBGL/data_ours/bugs/bugs_dev_generated.json \
    --test_file src/models/HBGL/data_ours/bugs/bugs_test_generated.json \
    --add_vocab_file src/models/HBGL/data_ours/bugs/label_map.pkl \
    --label_smoothing 0 \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps 96000 --cache_dir ${CACHE_DIR}\
    --random_prob 0 --keep_prob 0 --soft_label --seed ${seed} \
    --label_cpt src/models/HBGL/data_ours/bugs/bugs.taxnomy --label_cpt_not_incr_mask_ratio --label_cpt_steps 300 --label_cpt_use_bce \
    --wandb \
    --taxonomy_file data/Bugs/bugs_tax.txt
#    --only_test --only_test_path src/models/HBGL/dumps/bugs/ckpt-69000

    # comment the last line ("only_test") to use in training
