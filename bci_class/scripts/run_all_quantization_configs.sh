#!/bin/bash

# Script to run quantization for all possible configurations
# This script tests all 4 quantization methods with their configurations

set -e  # Exit on error

# Default values
MODEL_PATH=""
DATA_DIR="./data/hdf5_data_final"
CSV_PATH="./data/t15_copyTaskData_description.csv"
GPU_NUMBER=0
EVAL_TYPE="val"
SKIP_LM=false
OUTPUT_BASE_DIR=""
VERBOSE=false

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

Run quantization for all possible configurations (4 methods).

Required:
    --model_path PATH          Path to the pretrained model directory

Optional:
    --data_dir PATH            Dataset directory (default: ./data/hdf5_data_final)
    --csv_path PATH            CSV file with metadata (default: ./data/t15_copyTaskData_description.csv)
    --gpu_number N              GPU number to use, -1 for CPU (default: 0)
    --eval_type TYPE           Evaluation type: val or test (default: val)
    --output_base_dir PATH     Base directory for outputs (default: same as model_path parent)
    --skip_lm                  Skip language model inference for faster evaluation
    --verbose                  Print verbose output
    -h, --help                 Show this help message

Quantization Methods Tested:
    1. Dynamic INT8 (qint8)
    2. Dynamic FP16 (float16)
    3. Hybrid Dynamic+Static Per-Tensor INT8 (qint8)
    4. Hybrid Dynamic+Per-Channel Static INT8 (qint8)

Example:
    $0 --model_path ./data/t15_pretrained_rnn_baseline --gpu_number 0 --skip_lm
EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
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
        --output_base_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --skip_lm)
            SKIP_LM=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
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
if [[ -z "$MODEL_PATH" ]]; then
    print_error "Model path is required!"
    usage
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    print_error "Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Set output base directory
if [[ -z "$OUTPUT_BASE_DIR" ]]; then
    OUTPUT_BASE_DIR=$(dirname "$MODEL_PATH")
fi

# Get script directory (assuming script is in bci_class/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root directory and set PYTHONPATH so Python can find modules
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/model_training:$PYTHONPATH"

# Find Python executable from conda environment
PYTHON_CMD="python"
if command -v conda &> /dev/null; then
    # Try to source conda.sh and activate b2txt25 environment
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        # Activate the conda environment
        if conda env list | grep -q "^b2txt25 "; then
            print_info "Activating conda environment b2txt25..."
            conda activate b2txt25
            # Try to get Python from the conda environment
            CONDA_PYTHON="$(conda info --base)/envs/b2txt25/bin/python"
            if [[ -f "$CONDA_PYTHON" ]]; then
                PYTHON_CMD="$CONDA_PYTHON"
                print_info "Using Python from conda environment b2txt25: $PYTHON_CMD"
            else
                print_warning "Conda environment b2txt25 found but Python not found at $CONDA_PYTHON"
                print_warning "Falling back to system Python. Make sure torch is installed."
            fi
        else
            print_warning "Conda environment 'b2txt25' not found."
            print_warning "Falling back to system Python. Make sure torch is installed."
        fi
    fi
else
    print_warning "Conda not found. Using system Python. Make sure torch is installed."
fi

# Check if Python script exists (paths relative to project root after cd)
QUANTIZE_SCRIPT="$PROJECT_ROOT/bci_class/scripts/quantize_model.py"
EVAL_SCRIPT="$PROJECT_ROOT/bci_class/scripts/evaluate_quantized_model.py"

if [[ ! -f "$QUANTIZE_SCRIPT" ]]; then
    print_error "Quantization script not found: $QUANTIZE_SCRIPT"
    exit 1
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
    print_error "Evaluation script not found: $EVAL_SCRIPT"
    exit 1
fi

# Define all quantization configurations
declare -a CONFIGS=(
    "dynamic:qint8:Method 1: Dynamic INT8"
    "dynamic:float16:Method 2: Dynamic FP16"
    "static:qint8:Method 3: Hybrid Dynamic+Static Per-Tensor INT8"
    "static_per_channel:qint8:Method 4: Hybrid Dynamic+Per-Channel Static INT8"
)

# Create results directory (fixed location, no timestamp)
RESULTS_DIR="$OUTPUT_BASE_DIR/quantization_results"
mkdir -p "$RESULTS_DIR"

print_info "=========================================="
print_info "Running All Quantization Configurations"
print_info "=========================================="
print_info "Model path: $MODEL_PATH"
print_info "Data dir: $DATA_DIR"
print_info "CSV path: $CSV_PATH"
print_info "GPU number: $GPU_NUMBER"
print_info "Eval type: $EVAL_TYPE"
print_info "Output base: $OUTPUT_BASE_DIR"
print_info "Results dir: $RESULTS_DIR"
print_info "Total configurations: ${#CONFIGS[@]}"
print_info "=========================================="
echo ""

# Arrays to store results
declare -a SUCCESSFUL_METHODS=()
declare -a FAILED_METHODS=()
declare -a SKIPPED_METHODS=()

# Function to check if a configuration is already completed
is_config_completed() {
    local qtype=$1
    local dtype=$2
    local output_dir="$OUTPUT_BASE_DIR/quantized_${qtype}_${dtype}"
    local model_output_path="$output_dir/checkpoint/quantized_model.pt"
    local quantize_log="$RESULTS_DIR/${qtype}_${dtype}_quantize.log"
    local eval_log="$RESULTS_DIR/${qtype}_${dtype}_eval.log"
    
    # Check if quantized model exists
    if [[ ! -f "$model_output_path" ]]; then
        return 1  # Not completed
    fi
    
    # Check if quantize log exists and indicates success (no obvious errors)
    if [[ ! -f "$quantize_log" ]]; then
        return 1  # Not completed
    fi
    
    # Check if eval log exists and contains PER (indicates successful evaluation)
    if [[ ! -f "$eval_log" ]]; then
        return 1  # Not completed
    fi
    
    # Check if eval log contains "Phoneme Error Rate" which indicates successful completion
    if ! grep -qi "Phoneme Error Rate" "$eval_log" > /dev/null 2>&1; then
        return 1  # Not completed
    fi
    
    return 0  # Completed
}

# Function to run quantization and evaluation for a configuration
run_config() {
    local qtype=$1
    local dtype=$2
    local method_name=$3
    local config_num=$4
    local total_configs=$5
    
    print_info "[$config_num/$total_configs] Processing: $method_name"
    print_info "  Type: $qtype, Dtype: $dtype"
    
    # Check if this configuration is already completed
    if is_config_completed "$qtype" "$dtype"; then
        print_warning "  Configuration already completed, skipping..."
        
        # Try to extract PER from existing log
        local eval_log="$RESULTS_DIR/${qtype}_${dtype}_eval.log"
        if [[ -f "$eval_log" ]]; then
            local per=$(grep -i "Phoneme Error Rate" "$eval_log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
            if [[ -n "$per" ]]; then
                print_info "  PER: $per (from previous run)"
            fi
        fi
        
        SKIPPED_METHODS+=("$method_name")
        SUCCESSFUL_METHODS+=("$method_name")
        return 0
    fi
    
    # Create output directory for this configuration
    local output_dir="$OUTPUT_BASE_DIR/quantized_${qtype}_${dtype}"
    local model_output_path="$output_dir/checkpoint/quantized_model.pt"
    
    # Check if quantized model already exists and is valid
    if [[ -f "$model_output_path" ]]; then
        # Try to verify it's a valid quantized model by checking if Python can load it
        local check_cmd=(
            "$PYTHON_CMD" -c "
import torch
import sys
try:
    checkpoint = torch.load('$model_output_path', weights_only=False, map_location='cpu')
    if checkpoint.get('quantized', False):
        qtype = checkpoint.get('quantization_type', '')
        dtype = checkpoint.get('dtype', '')
        if qtype == '$qtype' and dtype == '$dtype':
            state_dict = checkpoint.get('model_state_dict', {})
            if state_dict:
                print('VALID')
                sys.exit(0)
    sys.exit(1)
except Exception as e:
    sys.exit(1)
"
        )
        
        if "${check_cmd[@]}" > /dev/null 2>&1; then
            print_warning "  Quantized model already exists and is valid, skipping quantization..."
            print_info "  Using existing model at: $model_output_path"
        else
            print_warning "  Quantized model exists but appears invalid, re-quantizing..."
            # Remove the invalid model file
            rm -f "$model_output_path"
        fi
    fi
    
    # Step 1: Quantize (only if model doesn't exist or is invalid)
    if [[ ! -f "$model_output_path" ]]; then
        print_info "  Step 1: Quantizing model..."
        local quantize_cmd=(
            "$PYTHON_CMD" "$QUANTIZE_SCRIPT"
            --model_path "$MODEL_PATH"
            --output_path "$model_output_path"
            --quantization_type "$qtype"
            --dtype "$dtype"
            --data_dir "$DATA_DIR"
            --csv_path "$CSV_PATH"
            --gpu_number "$GPU_NUMBER"
        )
        
        # Add calibration parameters for static methods
        if [[ "$qtype" == "static" ]] || [[ "$qtype" == "static_per_channel" ]]; then
            quantize_cmd+=(
                --calibration_batch_size 32
                --calibration_samples 1000
            )
        fi
        
        if [[ "$VERBOSE" == true ]]; then
            print_info "  Command: ${quantize_cmd[*]}"
        fi
        
        local quantize_start=$(date +%s)
        if "${quantize_cmd[@]}" > "$RESULTS_DIR/${qtype}_${dtype}_quantize.log" 2>&1; then
            local quantize_end=$(date +%s)
            local quantize_time=$((quantize_end - quantize_start))
            print_success "  Quantization completed in ${quantize_time}s"
        else
            print_error "  Quantization failed! Check log: $RESULTS_DIR/${qtype}_${dtype}_quantize.log"
            FAILED_METHODS+=("$method_name (quantization failed)")
            return 1
        fi
    else
        print_info "  Step 1: Skipped (model already exists)"
    fi
    
    # Step 2: Evaluate
    print_info "  Step 2: Evaluating quantized model..."
    local eval_cmd=(
        "$PYTHON_CMD" "$EVAL_SCRIPT"
        --model_path "$model_output_path"
        --data_dir "$DATA_DIR"
        --eval_type "$EVAL_TYPE"
        --csv_path "$CSV_PATH"
        --gpu_number "$GPU_NUMBER"
    )
    
    if [[ "$SKIP_LM" == true ]]; then
        eval_cmd+=(--skip_lm)
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        print_info "  Command: ${eval_cmd[*]}"
    fi
    
    local eval_start=$(date +%s)
    if "${eval_cmd[@]}" > "$RESULTS_DIR/${qtype}_${dtype}_eval.log" 2>&1; then
        local eval_end=$(date +%s)
        local eval_time=$((eval_end - eval_start))
        print_success "  Evaluation completed in ${eval_time}s"
        
        # Try to extract PER from log
        if [[ -f "$RESULTS_DIR/${qtype}_${dtype}_eval.log" ]]; then
            local per=$(grep -i "Phoneme Error Rate" "$RESULTS_DIR/${qtype}_${dtype}_eval.log" | grep -oE "[0-9]+\.[0-9]+" | head -1)
            if [[ -n "$per" ]]; then
                print_info "  PER: $per"
            fi
        fi
        
        SUCCESSFUL_METHODS+=("$method_name")
        return 0
    else
        print_error "  Evaluation failed! Check log: $RESULTS_DIR/${qtype}_${dtype}_eval.log"
        FAILED_METHODS+=("$method_name (evaluation failed)")
        return 1
    fi
}

# Run all configurations
TOTAL_START=$(date +%s)
CONFIG_NUM=0

for config in "${CONFIGS[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))
    IFS=':' read -r qtype dtype method_name <<< "$config"
    
    echo ""
    run_config "$qtype" "$dtype" "$method_name" "$CONFIG_NUM" "${#CONFIGS[@]}"
    
    # Small delay between configurations
    sleep 2
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

# Print summary
echo ""
print_info "=========================================="
print_info "Summary"
print_info "=========================================="
print_info "Total time: ${TOTAL_TIME}s"
print_info "Successful: ${#SUCCESSFUL_METHODS[@]}/${#CONFIGS[@]}"
print_info "Failed: ${#FAILED_METHODS[@]}/${#CONFIGS[@]}"
if [[ ${#SKIPPED_METHODS[@]} -gt 0 ]]; then
    print_info "Skipped (already completed): ${#SKIPPED_METHODS[@]}/${#CONFIGS[@]}"
fi
echo ""

if [[ ${#SUCCESSFUL_METHODS[@]} -gt 0 ]]; then
    print_success "Successful methods:"
    for method in "${SUCCESSFUL_METHODS[@]}"; do
        echo "  ✓ $method"
    done
    echo ""
fi

if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
    print_error "Failed methods:"
    for method in "${FAILED_METHODS[@]}"; do
        echo "  ✗ $method"
    done
    echo ""
fi

if [[ ${#SKIPPED_METHODS[@]} -gt 0 ]]; then
    print_warning "Skipped methods (already completed):"
    for method in "${SKIPPED_METHODS[@]}"; do
        echo "  ⊘ $method"
    done
    echo ""
fi

print_info "Results directory: $RESULTS_DIR"
print_info "Quantized models saved in: $OUTPUT_BASE_DIR/quantized_*"
print_info "=========================================="

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
cat > "$SUMMARY_FILE" << EOF
Quantization Results Summary
============================
Date: $(date)
Model: $MODEL_PATH
Total configurations: ${#CONFIGS[@]}
Total time: ${TOTAL_TIME}s

Successful methods (${#SUCCESSFUL_METHODS[@]}):
$(for method in "${SUCCESSFUL_METHODS[@]}"; do echo "  ✓ $method"; done)

Failed methods (${#FAILED_METHODS[@]}):
$(for method in "${FAILED_METHODS[@]}"; do echo "  ✗ $method"; done)

Skipped methods (${#SKIPPED_METHODS[@]}):
$(for method in "${SKIPPED_METHODS[@]}"; do echo "  ⊘ $method"; done)

Configuration details:
$(for config in "${CONFIGS[@]}"; do echo "  - $config"; done)

Log files are in: $RESULTS_DIR
Quantized models are in: $OUTPUT_BASE_DIR/quantized_*
EOF

print_success "Summary saved to: $SUMMARY_FILE"

# Exit with error if any method failed
if [[ ${#FAILED_METHODS[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi

