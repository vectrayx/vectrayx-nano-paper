#!/bin/bash
# Launch Base 260M LoRA con corpus mini_v1 — ON-DEMAND

set -e
export AWS_DEFAULT_REGION=us-east-1
TIMESTAMP=$(date +%Y%m%d-%H%M)
JOB_NAME="vectrayx-lora-base-mini-s42-${TIMESTAMP}"

echo "[job] ${JOB_NAME}"
echo "[model] Base 260M LoRA rank=16"
echo "[corpus] tool_sft_mini_v1 (2801 ejemplos, ratio 1:21)"
echo "[mode] ON-DEMAND"

aws sagemaker create-training-job \
  --training-job-name "${JOB_NAME}" \
  --role-arn "arn:aws:iam::792811916323:role/VectrayxSageMakerRole" \
  --algorithm-specification '{
    "TrainingImage": "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04",
    "TrainingInputMode": "File"
  }' \
  --hyper-parameters '{
    "sagemaker_program": "aws_lora_base_tools_s3.py",
    "sagemaker_submit_directory": "s3://vectrayx-sagemaker-792811916323/code/lora_nano_s3.tar.gz"
  }' \
  --output-data-config '{
    "S3OutputPath": "s3://vectrayx-sagemaker-792811916323/output/'${JOB_NAME}'",
    "CompressionType": "GZIP"
  }' \
  --resource-config '{
    "InstanceType": "ml.g5.xlarge",
    "InstanceCount": 1,
    "VolumeSizeInGB": 100
  }' \
  --stopping-condition '{
    "MaxRuntimeInSeconds": 10800
  }' \
  --environment '{
    "CORPUS_NAME": "mini_v1",
    "EPOCHS": "5",
    "LR": "2e-4",
    "LORA_RANK": "16",
    "LORA_ALPHA": "32",
    "SEED": "42"
  }'

echo "[s3 out] s3://vectrayx-sagemaker-792811916323/output/${JOB_NAME}"
