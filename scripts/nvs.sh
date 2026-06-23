#! /bin/bash

# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 
# Usage
# 
# 2 GPUs Training
# bash ./scripts/nvs.sh --ray_encoding camray --pos_enc urope --gpus "0,1" --dataset_name realestate10k
# 
# 2 GPUs Testing (with zooming in)
# bash ./scripts/nvs.sh --ray_encoding camray --pos_enc urope --gpus "0,1" --test-zoom-in "1 3 5" --dataset_name realestate10k
#
# 2 GPUs Testing (with more context views)
# bash ./scripts/nvs.sh --ray_encoding camray --pos_enc urope --gpus "0,1" --test-context-views "2 4 8 16" --dataset_name realestate10k
#
# 2 GPUs Testing (with rendering video)
# bash ./scripts/nvs.sh --ray_encoding camray --pos_enc urope --gpus "0,1" --test-render-video --dataset_name realestate10k


# Parse command line arguments
DATASET_NAME="realestate10k" # realestate10k or objaverse

while [[ $# -gt 0 ]]; do
  case $1 in
    --ray_encoding)
      RAY_ENCODING="$2"   
      shift 2
      ;;
    --pos_enc)
      POS_ENC="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --test-zoom-in)
      TEST_ZOOM_IN="$2"
      shift 2
      ;;
    --test-context-views)
      TEST_CONTEXT_VIEWS="$2"
      shift 2
      ;;
    --dataset_name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --test-render-video)
      TEST_RENDER_VIDEO=true
      shift 1
      ;;
    -h|--help)
      echo "Usage: $0 --ray_encoding <ray_encoding> --pos_enc <pos_enc> --gpus <gpu_list> [--test-zoom-in <zoom_factors>]"
      echo "  --ray_encoding: plucker, camray, none, or raymap"
      echo "  --pos_enc: urope, prope, gta, rope2d, or none"
      echo "  --gpus: comma-separated GPU list (e.g., '0,1')"
      echo "  --test-zoom-in: space-separated zoom factors for testing (e.g., '3 5')"
      echo "  --test-context-views: space-separated context views for testing (e.g., '2 4 8 16')"
      echo "  --test-render-video: render video for testing"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# URoPE parameters
if [ "$DATASET_NAME" == "realestate10k" ]; then
  URoPE_MIN_DEPTH=2.0
  URoPE_MAX_DEPTH=20.0
elif [ "$DATASET_NAME" == "objaverse" ]; then
  URoPE_MIN_DEPTH=0.7
  URoPE_MAX_DEPTH=2.6
fi

# Check required arguments
if [ -z "$RAY_ENCODING" ]; then
  echo "Error: --ray_encoding is required"
  exit 1
fi

if [ -z "$POS_ENC" ]; then
  echo "Error: --pos_enc is required"
  exit 1
fi

if [ -z "$GPUS" ]; then
  echo "Error: --gpus is required"
  exit 1
fi


NGPUS=$(echo $GPUS | tr ',' '\n' | wc -l)

NAME="release-${NGPUS}gpus-b8-s1-80k"
BASE_CMD=(
    "NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS"
    "nvs/trainval.py lvsm"
    "--amp --amp_dtype fp16"
    "--dataset_batch_scenes 8"
    "--dataset_supervise_views 1"
    "--model_config.encoder.num_layers 6"
    "--model_config.encoder.layer.d_model 768"
    "--model_config.encoder.layer.nhead 16"
    "--model_config.encoder.layer.dim_feedforward 1024"
    "--model_config.encoder.layer.qk_norm"
    "--model_config.urope_min_depth ${URoPE_MIN_DEPTH}"
    "--model_config.urope_max_depth ${URoPE_MAX_DEPTH}"
    "--max_steps 80000 --test_every 8000"
    "--model_config.ray_encoding ${RAY_ENCODING}"
    "--model_config.pos_enc ${POS_ENC}"
    "--output_dir results/outputs-${DATASET_NAME}/${NAME}-${RAY_ENCODING}-${POS_ENC}"
    "--dataset_name ${DATASET_NAME}"
)

echo "NAME: ${NAME}"
echo "RAY_ENCODING: ${RAY_ENCODING}"
echo "POS_ENC: ${POS_ENC}"

if [ -n "$TEST_ZOOM_IN" ]; then
    for zoom_factor in $TEST_ZOOM_IN; do
        echo "Starting testing with zoom factor ${zoom_factor}..."
        CMD=(
            "${BASE_CMD[@]}"
            "--test_only --auto_resume"
            "--test_zoom_factor ${zoom_factor}"
            "--test_subdir eval-zoom${zoom_factor}x"
        )
        CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    done
    exit 0
elif [ -n "$TEST_CONTEXT_VIEWS" ]; then
    for context_views in $TEST_CONTEXT_VIEWS; do
        echo "Starting testing with ${context_views} context views..."
        CMD=(
            "${BASE_CMD[@]}"
            "--test_only --auto_resume"
            "--model_config.ref_views ${context_views}"
            "--test_input_views ${context_views}"
            "--test_index_fp evaluation_index_re10k_context${context_views}.json"
            "--test_subdir eval-context${context_views}"
        )
        CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    done
    exit 0
elif [ -n "$TEST_RENDER_VIDEO" ]; then
    echo "Starting testing with rendering video for fisrt 8 scenes ..."
    CMD=(
        "${BASE_CMD[@]}"
        "--test_only --auto_resume --render_video --test_n 8"
    )
    CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    exit 0
else
    echo "Starting training process..."
    CMD=(
        "${BASE_CMD[@]}"
    )
    CUDA_VISIBLE_DEVICES=$GPUS eval "${CMD[@]}"
    exit 0
fi