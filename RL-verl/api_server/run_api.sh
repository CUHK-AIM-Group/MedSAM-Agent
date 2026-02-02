#!/bin/bash

set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-medsam2}  # imisnet | medsam2
PORT=${PORT:-8265}

if [[ "$MODEL_TYPE" == "imisnet" ]]; then
  export MODEL_TYPE="imisnet"
  export IMISNET_CHECKPOINT=${IMISNET_CHECKPOINT:-/your/path/to/IMISNet-B.pth}
  export IMISNET_IMAGE_SIZE=${IMISNET_IMAGE_SIZE:-1024}
else
  export MODEL_TYPE="medsam2"
  export MEDSAM2_CHECKPOINT=${MEDSAM2_CHECKPOINT:-/your/path/to/MedSAM2_latest.pt}
  export MEDSAM2_CONFIG=${MEDSAM2_CONFIG:-configs/sam2.1/sam2.1_hiera_t.yaml}
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$SCRIPT_DIR/segmentation_api.py"
