#!/bin/bash

# Full model path (directory for local model, or "gpt" for API mode)
MODEL_PATH="/path/to/your/model"
# Segmentation model configuration (still required for GPT mode for segmentation)
SEG_MODEL="medsam"  # Segmentation model type: sam, medsam, imisnet

SAVE_INTERMEDIATE="true"  # Save intermediate_results (true/false)

DATASETS=("LiverUS")
SPLIT="test"

# SAM/MedSAM model paths and config (used for all modes)
# SAM2
SAM2_CHECKPOINT="path/models--facebook--sam2.1-hiera-base-plus/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_b+.yaml"

# MedSAM2
MEDSAM2_CHECKPOINT="path/models--wanglab--MedSAM2/MedSAM2_latest.pt"
MEDSAM2_CONFIG="configs/sam2.1/sam2.1_hiera_t.yaml"

# IMISNet checkpoint path
IMISNET_CHECKPOINT="path/models--1Junlong--IMIS-Net/IMISNet-B.pth"
# === Data path configuration ===
DATA_ROOT="your/data/root"

# Select checkpoint/config based on SEG_MODEL
if [ "$SEG_MODEL" = "medsam" ]; then
    SEG_CHECKPOINT="$MEDSAM2_CHECKPOINT"
    SEG_CONFIG="$MEDSAM2_CONFIG"
elif [ "$SEG_MODEL" = "imisnet" ]; then
    SEG_CHECKPOINT="$IMISNET_CHECKPOINT"
    SEG_CONFIG=""  # Add config path if needed
else
    SEG_CHECKPOINT="$SAM2_CHECKPOINT"
    SEG_CONFIG="$SAM2_CONFIG"
fi

# Other common configs
N_CLICKS=5
N_GPUS=8
PROCESSES_PER_GPU=2  # Processes per GPU

# Performance options
USE_FP16="true"  # Use FP16 mixed precision
BATCH_SIZE=1     # Batch size (currently only supports 1)

# Auto-expand datasets with one-level subfolders
expand_datasets() {
    local input_datasets=("$@")
    local expanded=()
    local split_folders=("train" "test" "val" "train_mask" "test_mask" "val_mask")

    for dataset in "${input_datasets[@]}"; do
        if [[ "$dataset" == */* ]]; then
            expanded+=("$dataset")
            continue
        fi

        local dataset_path="${DATA_ROOT}/${dataset}"
        if [ ! -d "$dataset_path" ]; then
            echo "[WARN] Dataset directory not found, using as-is: ${dataset}" >&2
            expanded+=("$dataset")
            continue
        fi

        local discovered=()
        while IFS= read -r -d '' sub_dir; do
            local sub_name=$(basename "$sub_dir")
            if [[ "$sub_name" == "__pycache__" || "$sub_name" == .* ]]; then
                continue
            fi

            local is_split=false
            for split in "${split_folders[@]}"; do
                if [[ "$sub_name" == "$split" ]]; then
                    is_split=true
                    break
                fi
            done
            if [ "$is_split" = true ]; then
                continue
            fi

            local has_splits=false
            for split in "train" "test" "val"; do
                if [ -d "${sub_dir}/${split}" ]; then
                    has_splits=true
                    break
                fi
            done
            if [ "$has_splits" = true ]; then
                discovered+=("$sub_name")
            fi
        done < <(find "$dataset_path" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)

        if [ ${#discovered[@]} -gt 0 ]; then
            for sub_name in "${discovered[@]}"; do
                expanded+=("${dataset}/${sub_name}")
            done
        else
            expanded+=("$dataset")
        fi
    done

    printf '%s\n' "${expanded[@]}"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --datasets   Comma-separated dataset list (default: ${DATASETS[*]})"
    echo "  -s, --split      Dataset split to evaluate (default: $SPLIT)"
    echo "  -h, --help       Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--datasets)
            if [[ -z ${2:-} ]]; then
                echo "Error: --datasets expects a value"
                exit 1
            fi
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        -s|--split)
            if [[ -z ${2:-} ]]; then
                echo "Error: --split expects a value"
                exit 1
            fi
            SPLIT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

mapfile -t DATASETS < <(expand_datasets "${DATASETS[@]}")
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: No datasets to process after expansion"
    exit 1
fi

# ============================================
# Script start
# ============================================

echo "=========================================="
echo "Batch Multi-GPU Inference Job"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Datasets: ${DATASETS[*]}"
echo "Number of GPUs: $N_GPUS"
echo "Processes per GPU: $PROCESSES_PER_GPU"
echo "Total parallel processes: $((N_GPUS * PROCESSES_PER_GPU))"
echo "=========================================="
echo ""

# Record total start time
TOTAL_START_TIME=$(date +%s)

# Set Python environment variables once
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export VIS_DIR="$(pwd)"
# Validate model path (skip for GPT mode)
if [ "$MODEL_PATH" != "gpt" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Run for all datasets
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing Dataset: $DATASET_NAME"
    echo "=========================================="
    echo ""

    DATASET_START_TIME=$(date +%s)

    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="./intermediate_result/${MODEL_NAME}/${DATASET_NAME}"
    RESULTS_DIR="./results/${MODEL_NAME}-${SEG_MODEL}/${DATASET_NAME}"

    mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

    echo "Configuration:"
    echo "  Dataset: $DATASET_NAME"
    echo "  Model path: $MODEL_PATH"
    echo "  Intermediate results dir: $OUTPUT_DIR"
    echo "  Summary results dir: $RESULTS_DIR"
    echo "=========================================="

    RESULTS_FILE="$RESULTS_DIR/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Results file already exists: $RESULTS_FILE"
        echo "Skipping dataset $DATASET_NAME"
        echo ""
        continue
    fi

    if [ $N_GPUS -eq 1 ]; then
        export CUDA_VISIBLE_DEVICES=0
    elif [ $N_GPUS -eq 2 ]; then
        export CUDA_VISIBLE_DEVICES=0,1
    elif [ $N_GPUS -eq 3 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2
    elif [ $N_GPUS -eq 4 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
    elif [ $N_GPUS -eq 5 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4
    elif [ $N_GPUS -eq 6 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    elif [ $N_GPUS -eq 7 ]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
    else
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    fi

    echo "Visible GPU devices: $CUDA_VISIBLE_DEVICES"

    LOG_FILE="$RESULTS_DIR/inference.log"
    echo "Saving logs to: $LOG_FILE"
    echo "Running inference (detailed logs in $LOG_FILE)..."

    PYTHON_ARGS=(
        "--dataset_name" "$DATASET_NAME"
        "--split" "$SPLIT"
        "--model_path" "$MODEL_PATH"
        "--save_intermediate" "$SAVE_INTERMEDIATE"
        "--n_clicks" "$N_CLICKS"
        "--n_gpus" "$N_GPUS"
        "--processes_per_gpu" "$PROCESSES_PER_GPU"
        "--data_root" "$DATA_ROOT"
        "--output_dir" "$OUTPUT_DIR"
        "--results_dir" "$RESULTS_DIR"
        "--seg_model" "$SEG_MODEL"
        "--seg-checkpoint" "$SEG_CHECKPOINT"
        "--seg-config" "$SEG_CONFIG"
        "--use_fp16" "$USE_FP16"
        "--batch_size" "$BATCH_SIZE"
    )

    python unibiomed_inference_toolusing_multi_gpu.py "${PYTHON_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "Inference completed for dataset $DATASET_NAME"

        if [ "$SAVE_INTERMEDIATE" = "false" ]; then
            echo "Cleaning up intermediate results directory..."
            if [ -d "$OUTPUT_DIR" ]; then
                rm -rf "$OUTPUT_DIR"
                echo "Intermediate results directory removed: $OUTPUT_DIR"
            fi
        fi

        DATASET_END_TIME=$(date +%s)
        DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
        echo "Elapsed time: $((DATASET_DURATION / 60)) minutes $((DATASET_DURATION % 60)) seconds"

        echo "Results saved at:"
        echo "  - Intermediate results: $OUTPUT_DIR"
        echo "  - Summary results: $RESULTS_DIR/results.json"
        echo "  - Inference log: $RESULTS_DIR/inference.log"

        echo "Cleaning cache after dataset $DATASET_NAME..."
        python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
        python -c "import gc; gc.collect(); print('Python garbage collected')" 2>/dev/null || true
        sync
        echo "Cache cleaned successfully."
    else
        echo "Error: Inference failed for dataset $DATASET_NAME"
    fi

    echo "Completed dataset: $DATASET_NAME"
    echo "=========================================="
    echo ""
done

# Total elapsed time
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "=========================================="
echo "All inference tasks completed!"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo "Total time: $((TOTAL_DURATION / 3600)) hours $(((TOTAL_DURATION % 3600) / 60)) minutes $((TOTAL_DURATION % 60)) seconds"
echo "=========================================="
echo ""
echo "Results summary:"
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "  Dataset $DATASET_NAME:"
    MODEL_NAME=$(basename "$MODEL_PATH")
    RESULTS_DIR="./results/${MODEL_NAME}/${DATASET_NAME}"
    RESULTS_FILE="${RESULTS_DIR}/results.json"
    LOG_FILE="${RESULTS_DIR}/inference.log"
    if [ -f "$RESULTS_FILE" ]; then
        echo "    - Results: $RESULTS_FILE"
        if [ -f "$LOG_FILE" ]; then
            echo "    - Log: $LOG_FILE"
        fi
    fi
done