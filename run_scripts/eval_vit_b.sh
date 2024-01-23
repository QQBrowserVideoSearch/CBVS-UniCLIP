#!/usr/bin/env

test_data=datasets/cbvs20k.txt

# restore options
resume=pretrained/UniCLIP-base.pt

# inference output
output_file=output/inference_result.txt

# model config
ocr_presence=1
ocr_semantic=1

# test hyper-params
context_length=12
batch_size=160
input_resolution=224

# ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "RN50", "ViT-H-14"]
vision_model=ViT-B-16

# ["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]
text_model=RoBERTa-wwm-ext-base-chinese

export PYTHONPATH=${PYTHONPATH}:`pwd`/uniclip/

python3  uniclip/eval/main.py \
          --test-data=${test_data} \
          --resume=${resume} \
          --output-file=${output_file} \
          --context-length=${context_length} \
          --batch-size=${batch_size} \
          --input-resolution=${input_resolution} \
          --vision-model=${vision_model} \
          --text-model=${text_model} \
          --ocr-presence=${ocr_presence} \
          --ocr-semantic=${ocr_semantic} \