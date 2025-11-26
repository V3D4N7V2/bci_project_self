# Model Quantization

This module provides functionality to quantize RNN models for compression and faster inference.

## Overview

Quantization reduces the precision of model weights and activations, typically from 32-bit floats to 8-bit integers, resulting in:
- **Smaller model size** (typically 4x reduction)
- **Faster inference** (especially on CPU)
- **Lower memory usage**

## Supported Quantization Methods (4 Methods)

### Method 1: Dynamic Quantization (INT8)
- **Type**: `dynamic`, **Dtype**: `qint8`
- Quantizes weights to INT8 but computes activations in floating point
- Works well for RNNs without requiring calibration data
- Fast and simple, recommended for most use cases
- **Best for**: Quick deployment, CPU inference

### Method 2: Dynamic Quantization (FP16)
- **Type**: `dynamic`, **Dtype**: `float16`
- Quantizes weights to FP16 (half precision)
- Similar to INT8 but uses floating point representation
- Good for GPUs that support FP16
- **Best for**: GPU inference, when FP16 is supported

### Method 3: Static Quantization Per-Tensor (INT8)
- **Type**: `static`, **Dtype**: `qint8`
- Quantizes both weights and activations to INT8
- Uses per-tensor quantization (same scale/zero_point for entire tensor)
- Requires calibration data to determine quantization parameters
- Can provide better compression but more complex
- **Best for**: Maximum compression, when calibration data is available

### Method 4: Static Quantization Per-Channel (INT8)
- **Type**: `static_per_channel`, **Dtype**: `qint8`
- Quantizes both weights and activations to INT8
- Uses per-channel quantization (different scale/zero_point per channel)
- More accurate than per-tensor but slightly more complex
- Requires calibration data
- **Best for**: Best accuracy with static quantization

## Usage

### 1. Quantize a Model

```bash
python bci_class/scripts/quantize_model.py \
    --model_path ./data/t15_pretrained_rnn_baseline \
    --output_path ./data/quantized_model/checkpoint/quantized_model.pt \
    --quantization_type dynamic \
    --dtype qint8 \
    --gpu_number 0
```

### 2. Evaluate a Quantized Model

```bash
python bci_class/scripts/evaluate_quantized_model.py \
    --model_path ./data/quantized_model/checkpoint/quantized_model.pt \
    --data_dir ./data/hdf5_data_final \
    --eval_type val \
    --gpu_number 0
```

### 3. Test All 4 Methods (Recommended)

```bash
python bci_class/scripts/test_all_quantization_methods.py \
    --model_path ./data/t15_pretrained_rnn_baseline \
    --eval_type val \
    --gpu_number 0 \
    --skip_lm
```

This will test all 4 quantization methods and generate a comparison report.

### 4. Run Single Method Experiment (Quantize + Evaluate)

```bash
python bci_class/scripts/run_quantization_experiment.py \
    --model_path ./data/t15_pretrained_rnn_baseline \
    --quantization_type dynamic \
    --dtype qint8 \
    --eval_type val \
    --gpu_number 0
```

## Parameters

### Quantization Script (`quantize_model.py`)

- `--model_path`: Path to the pretrained model directory
- `--output_path`: Path to save the quantized model
- `--quantization_type`: `dynamic`, `static`, or `static_per_channel`
- `--dtype`: `qint8` (8-bit integer) or `float16` (16-bit float)
- `--data_dir`: Dataset directory (for static quantization calibration)
- `--csv_path`: CSV file with metadata
- `--gpu_number`: GPU number to use (-1 for CPU)
- `--calibration_batch_size`: Batch size for calibration (static only)
- `--calibration_samples`: Number of samples for calibration (static only)

### Evaluation Script (`evaluate_quantized_model.py`)

- `--model_path`: Path to the quantized model file
- `--data_dir`: Dataset directory
- `--eval_type`: `val` or `test`
- `--csv_path`: CSV file with metadata
- `--gpu_number`: GPU number to use (-1 for CPU)
- `--output_csv`: Path to save results CSV
- `--skip_lm`: Skip language model inference for faster evaluation

## Method Comparison

| Method | Type | Dtype | Calibration Needed | Accuracy | Speed | Best For |
|--------|------|-------|-------------------|----------|-------|----------|
| Method 1 | Dynamic | INT8 | No | Good | Fast | Quick deployment |
| Method 2 | Dynamic | FP16 | No | Good | Fast | GPU inference |
| Method 3 | Static | INT8 | Yes | Better | Fastest | Maximum compression |
| Method 4 | Static | INT8 | Yes | Best | Fastest | Best accuracy |

## Example Results

After quantization, you should see:
- Model size reduction (typically 3-4x for INT8, 2x for FP16)
- Similar or slightly reduced accuracy (Method 4 usually best)
- Faster inference speed (especially on CPU)
- Method 4 (per-channel) typically has best accuracy among static methods

## Notes

- **Method 1 (Dynamic INT8)** is recommended for quick deployment - simple and works well
- **Method 2 (Dynamic FP16)** is good for GPU inference when FP16 is supported
- **Method 3 & 4 (Static)** require calibration data but can provide better compression
- **Method 4 (Per-Channel)** typically has better accuracy than Method 3 (Per-Tensor)
- Quantized models may have slightly reduced accuracy compared to full-precision models
- For best results, test all 4 methods using `test_all_quantization_methods.py` and compare performance
- Use `--skip_lm` flag for faster evaluation (skips language model inference)

## Troubleshooting

1. **Import errors**: Make sure you're running from the project root directory
2. **CUDA errors**: Try using CPU (`--gpu_number -1`) if CUDA issues occur
3. **Memory errors**: Reduce batch size or calibration samples
4. **Static quantization issues**: Try dynamic quantization instead, which is more reliable for RNNs

