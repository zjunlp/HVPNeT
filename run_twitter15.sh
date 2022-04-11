#!/usr/bin/env bash
# Required environment variables:
# batch_size (recommendation: 8 / 16)
# lr: learning rate (recommendation: 3e-5 / 5e-5)
# seed: random seed, default is 1234
# BERT_NAME: pre-trained text model name ( bert-*)
# max_seq: max sequence length
# sample_ratio: few-shot learning, default is 1.0
# save_path: model saved path

DATASET_NAME="twitter15"
BERT_NAME="bert-base-uncased"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=30 \
        --batch_size=8 \
        --lr=$lr \
        --warmup_ratio=0.01 \
        --eval_begin_epoch=3 \
        --seed=1234 \
        --do_train \
        --ignore_idx=0 \
        --max_seq=128 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path=your_ckpt_path