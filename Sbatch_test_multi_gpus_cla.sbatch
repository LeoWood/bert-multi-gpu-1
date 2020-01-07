#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J test_tf_dis_4
#SBATCH -o test_tf_dis_4.out

export MIOPEN_USER_DB_PATH=/tmp/tensorflow-miopen-${USER}-2.8
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

python run_custom_classifier.py \
    --task_name=cla \
    --do_lower_case=true \
    --do_train=true \
    --do_predict=true \
    --save_for_serving=true \
    --data_dir=/public/home/zzx6320/lh/Projects/bert/data/data_wanfang_16000 \
    --vocab_file=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=400 \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=2.0 \
    --use_gpu=true \
    --num_gpu_cores=4 \
    --use_fp16=false \
    --output_dir=output/test_multi_gpus_cla \
    --cla_nums=16