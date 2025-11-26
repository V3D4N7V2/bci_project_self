#!/bin/bash

# Script to compare all quantized models against the original model
# This script finds all quantized models and compares them with the original

set -e  # Exit on error

# Default values
ORIGINAL_MODEL_PATH=""
QUANTIZED_BASE_DIR=""
DATA_DIR="./data/hdf5_data_final"
CSV_PATH="./data/t15_copyTaskData_description.csv"
GPU_NUMBER=0
EVAL_TYPE="val"
OUTPUT_DIR=""
NUM_SAMPLES=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Compare all quantized models against the original model.

Required:
    --original_model_path PATH    Path to the original (non-quantized) model directory
    --quantized_base_dir PATH     Base directory containing quantized models (e.g., ./data)

Optional:
    --data_dir PATH               Dataset directory (default: ./data/hdf5_data_final)
    --csv_path PATH               CSV file with metadata (default: ./data/t15_copyTaskData_description.csv)
    --gpu_number N                GPU number to use, -1 for CPU (default: 0)
    --eval_type TYPE              Evaluation type: val or test (default: val)
    --output_dir PATH             Directory to save comparison results (default: quantized_base_dir/comparison_results)
    --num_samples N               Limit number of samples for faster testing
    -h, --help                    Show this help message

The script will automatically find all quantized models in quantized_base_dir/quantized_* directories
and compare each against the original model.

Example:
    $0 --original_model_path ./data/t15_pretrained_rnn_baseline \\
       --quantized_base_dir ./data \\
       --gpu_number 0 \\
       --eval_type val
EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original_model_path)
            ORIGINAL_MODEL_PATH="$2"
            shift 2
            ;;
        --quantized_base_dir)
            QUANTIZED_BASE_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --gpu_number)
            GPU_NUMBER="$2"
            shift 2
            ;;
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [[ -z "$ORIGINAL_MODEL_PATH" ]]; then
    print_error "Original model path is required!"
    usage
fi

if [[ -z "$QUANTIZED_BASE_DIR" ]]; then
    print_error "Quantized base directory is required!"
    usage
fi

if [[ ! -d "$ORIGINAL_MODEL_PATH" ]]; then
    print_error "Original model path does not exist: $ORIGINAL_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$QUANTIZED_BASE_DIR" ]]; then
    print_error "Quantized base directory does not exist: $QUANTIZED_BASE_DIR"
    exit 1
fi

# Set output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$QUANTIZED_BASE_DIR/comparison_results"
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Find Python executable from conda environment
PYTHON_CMD="python"
if command -v conda &> /dev/null; then
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if conda env list | grep -q "^b2txt25 "; then
            CONDA_PYTHON="$(conda info --base)/envs/b2txt25/bin/python"
            if [[ -f "$CONDA_PYTHON" ]]; then
                PYTHON_CMD="$CONDA_PYTHON"
                print_info "Using Python from conda environment b2txt25: $PYTHON_CMD"
            fi
        fi
    fi
fi

# Change to project root directory and set PYTHONPATH
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/model_training:$PYTHONPATH"

# Check if comparison script exists
COMPARE_SCRIPT="$SCRIPT_DIR/compare_quantized_vs_original.py"
if [[ ! -f "$COMPARE_SCRIPT" ]]; then
    print_error "Comparison script not found: $COMPARE_SCRIPT"
    exit 1
fi

# Find all quantized model directories
print_info "Searching for quantized models in $QUANTIZED_BASE_DIR..."
QUANTIZED_MODELS=()
while IFS= read -r -d '' dir; do
    # Check if this directory contains a quantized model
    if [[ -f "$dir/checkpoint/quantized_model.pt" ]] || [[ -f "$dir/checkpoint/best_checkpoint" ]]; then
        QUANTIZED_MODELS+=("$dir")
        print_info "Found quantized model: $dir"
    fi
done < <(find "$QUANTIZED_BASE_DIR" -type d -name "quantized_*" -print0 2>/dev/null)

if [[ ${#QUANTIZED_MODELS[@]} -eq 0 ]]; then
    print_error "No quantized models found in $QUANTIZED_BASE_DIR"
    print_info "Looking for directories matching pattern: quantized_*"
    exit 1
fi

print_info "Found ${#QUANTIZED_MODELS[@]} quantized model(s) to compare"

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="$OUTPUT_DIR/comparison_results_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

print_info "=========================================="
print_info "Comparing Quantized Models vs Original"
print_info "=========================================="
print_info "Original model: $ORIGINAL_MODEL_PATH"
print_info "Quantized models: ${#QUANTIZED_MODELS[@]}"
print_info "Data dir: $DATA_DIR"
print_info "CSV path: $CSV_PATH"
print_info "GPU number: $GPU_NUMBER"
print_info "Eval type: $EVAL_TYPE"
print_info "Output dir: $RESULTS_DIR"
print_info "=========================================="
echo ""

# Arrays to store results
declare -a SUCCESSFUL_COMPARISONS=()
declare -a FAILED_COMPARISONS=()

# Function to run comparison
run_comparison() {
    local quantized_model=$1
    local model_num=$2
    local total_models=$3
    
    # Extract model name from path
    local model_name=$(basename "$quantized_model")
    print_info "[$model_num/$total_models] Comparing: $model_name"
    
    # Determine model path (could be directory or file)
    local model_path="$quantized_model"
    if [[ -d "$quantized_model" ]]; then
        if [[ -f "$quantized_model/checkpoint/quantized_model.pt" ]]; then
            model_path="$quantized_model/checkpoint/quantized_model.pt"
        elif [[ -f "$quantized_model/checkpoint/best_checkpoint" ]]; then
            model_path="$quantized_model"
        fi
    fi
    
    # Build command
    local compare_cmd=(
        "$PYTHON_CMD" "$COMPARE_SCRIPT"
        --original_model_path "$ORIGINAL_MODEL_PATH"
        --quantized_model_path "$model_path"
        --data_dir "$DATA_DIR"
        --csv_path "$CSV_PATH"
        --gpu_number "$GPU_NUMBER"
        --eval_type "$EVAL_TYPE"
        --output_csv "$RESULTS_DIR/${model_name}_comparison.csv"
    )
    
    if [[ -n "$NUM_SAMPLES" ]]; then
        compare_cmd+=(--num_samples "$NUM_SAMPLES")
    fi
    
    # Run comparison
    local start_time=$(date +%s)
    if "${compare_cmd[@]}" > "$RESULTS_DIR/${model_name}_comparison.log" 2>&1; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        print_success "  Comparison completed in ${elapsed}s"
        
        # Extract key metrics from log
        if [[ -f "$RESULTS_DIR/${model_name}_comparison.log" ]]; then
            local per_drop=$(grep -i "PER Increase" "$RESULTS_DIR/${model_name}_comparison.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
            local accuracy_drop=$(grep -i "Accuracy Drop" "$RESULTS_DIR/${model_name}_comparison.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
            if [[ -n "$per_drop" ]]; then
                print_info "  PER Increase: $per_drop"
            fi
            if [[ -n "$accuracy_drop" ]]; then
                print_info "  Accuracy Drop: $accuracy_drop"
            fi
        fi
        
        SUCCESSFUL_COMPARISONS+=("$model_name")
        return 0
    else
        print_error "  Comparison failed! Check log: $RESULTS_DIR/${model_name}_comparison.log"
        FAILED_COMPARISONS+=("$model_name")
        return 1
    fi
}

# Run all comparisons
TOTAL_START=$(date +%s)
MODEL_NUM=0

for quantized_model in "${QUANTIZED_MODELS[@]}"; do
    MODEL_NUM=$((MODEL_NUM + 1))
    echo ""
    run_comparison "$quantized_model" "$MODEL_NUM" "${#QUANTIZED_MODELS[@]}"
    sleep 1
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# Print summary
echo ""
print_info "=========================================="
print_info "Summary"
print_info "=========================================="
print_info "Total time: ${TOTAL_TIME}s"
print_info "Successful: ${#SUCCESSFUL_COMPARISONS[@]}/${#QUANTIZED_MODELS[@]}"
print_info "Failed: ${#FAILED_COMPARISONS[@]}/${#QUANTIZED_MODELS[@]}"
echo ""

if [[ ${#SUCCESSFUL_COMPARISONS[@]} -gt 0 ]]; then
    print_success "Successful comparisons:"
    for model in "${SUCCESSFUL_COMPARISONS[@]}"; do
        echo "  ✓ $model"
    done
    echo ""
fi

if [[ ${#FAILED_COMPARISONS[@]} -gt 0 ]]; then
    print_error "Failed comparisons:"
    for model in "${FAILED_COMPARISONS[@]}"; do
        echo "  ✗ $model"
    done
    echo ""
fi

print_info "Results directory: $RESULTS_DIR"
print_info "Comparison CSVs: $RESULTS_DIR/*_comparison.csv"
print_info "=========================================="

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
cat > "$SUMMARY_FILE" << EOF
Quantization Comparison Summary
================================
Date: $(date)
Original Model: $ORIGINAL_MODEL_PATH
Total quantized models: ${#QUANTIZED_MODELS[@]}
Total time: ${TOTAL_TIME}s

Successful comparisons (${#SUCCESSFUL_COMPARISONS[@]}):
$(for model in "${SUCCESSFUL_COMPARISONS[@]}"; do echo "  ✓ $model"; done)

Failed comparisons (${#FAILED_COMPARISONS[@]}):
$(for model in "${FAILED_COMPARISONS[@]}"; do echo "  ✗ $model"; done)

Comparison results are in: $RESULTS_DIR
EOF

print_success "Summary saved to: $SUMMARY_FILE"

# Exit with error if any comparison failed
if [[ ${#FAILED_COMPARISONS[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi

