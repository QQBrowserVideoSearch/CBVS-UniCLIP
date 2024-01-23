#!/usr/bin/env

# Number of GPUs per GPU worker
GPUS_PER_NODE=8

# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=127.0.0.1

# The port for communication
export MASTER_PORT=8518

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

export PYTHONPATH=${PYTHONPATH}:`pwd`/uniclip/


train_data=datasets/cbvs5m.txt


# restore options
resume=pretrained/QA-CLIP-base.pt

reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"


# output options
output_base_dir=output/
name=exp_10m

# model config
ocr_presence=1
ocr_semantic=1

save_step_frequency=9999999
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"

# training hyper-params
context_length=12
warmup=100
batch_size=180
valid_batch_size=180
accum_freq=1
lr=2e-5
wd=0.001 # weight delay
max_epochs=20

# ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "RN50", "ViT-H-14"]
vision_model=ViT-B-16

# ["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]
text_model=RoBERTa-wwm-ext-base-chinese

use_augment="--use-augment"
num_workers=1

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} uniclip/training/main.py \
          --train-data=${train_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --accum-freq=${accum_freq} \
          --lr=${lr} \
          --wd=${wd} \
          --num-workers=${num_workers} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          --text-model=${text_model} \
          --ocr-presence=${ocr_presence} \
          --ocr-semantic=${ocr_semantic} \