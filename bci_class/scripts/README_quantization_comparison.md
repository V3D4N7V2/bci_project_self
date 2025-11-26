# Quantization Model Comparison Scripts

These scripts allow you to compare quantized models against the original model to measure accuracy drop **without requiring the language model**. All comparisons are done at the phoneme level.

## Scripts

1. **`compare_quantized_vs_original.py`** - Compares a single quantized model against the original
2. **`compare_all_quantized_models.sh`** - Batch script to compare all quantized models found in a directory

## Features

- **No Language Model Required**: All comparisons are at the phoneme level, avoiding RAM-intensive language model inference
- **Multiple Metrics**: 
  - Logit-level comparison (MSE, MAE, cosine similarity)
  - Phoneme Error Rate (PER) comparison
  - Per-trial detailed comparisons
  - Accuracy drop calculation
- **Speed Comparison**: Measures inference speedup from quantization
- **Detailed Output**: Saves per-trial results to CSV for further analysis

## Usage

### Compare a Single Quantized Model

```bash
python bci_class/scripts/compare_quantized_vs_original.py \
    --original_model_path ./data/t15_pretrained_rnn_baseline \
    --quantized_model_path ./data/quantized_dynamic_qint8/checkpoint/quantized_model.pt \
    --data_dir ./data/hdf5_data_final \
    --csv_path ./data/t15_copyTaskData_description.csv \
    --gpu_number 0 \
    --eval_type val
```

### Compare All Quantized Models (Batch)

```bash
bash bci_class/scripts/compare_all_quantized_models.sh \
    --original_model_path ./data/t15_pretrained_rnn_baseline \
    --quantized_base_dir ./data \
    --gpu_number 0 \
    --eval_type val
```

This will automatically find all `quantized_*` directories and compare each against the original.

### Options

**`compare_quantized_vs_original.py` options:**
- `--original_model_path`: Path to original (non-quantized) model directory (required)
- `--quantized_model_path`: Path to quantized model file or directory (required)
- `--data_dir`: Dataset directory (default: `./data/hdf5_data_final`)
- `--csv_path`: CSV file with metadata (default: `./data/t15_copyTaskData_description.csv`)
- `--gpu_number`: GPU number to use, -1 for CPU (default: 0)
- `--eval_type`: Evaluation type: `val` or `test` (default: `val`)
- `--output_csv`: Path to save comparison results CSV (default: auto-generated)
- `--num_samples`: Limit number of samples for faster testing (optional)

**`compare_all_quantized_models.sh` options:**
- `--original_model_path`: Path to original model directory (required)
- `--quantized_base_dir`: Base directory containing quantized models (required)
- `--data_dir`: Dataset directory (default: `./data/hdf5_data_final`)
- `--csv_path`: CSV file with metadata (default: `./data/t15_copyTaskData_description.csv`)
- `--gpu_number`: GPU number to use, -1 for CPU (default: 0)
- `--eval_type`: Evaluation type: `val` or `test` (default: `val`)
- `--output_dir`: Directory to save comparison results (default: `quantized_base_dir/comparison_results`)
- `--num_samples`: Limit number of samples for faster testing (optional)

## Output

### Console Output

The script prints:
- Logit-level comparison metrics (MSE, MAE, cosine similarity)
- Phoneme Error Rate (PER) for both models
- PER increase (accuracy drop)
- Accuracy metrics
- Inference speed comparison

Example output:
```
Aggregate Metrics
============================================================

Logit-level Comparison:
  Average MSE: 0.000123
  Average MAE: 0.008456
  Average Cosine Similarity: 0.9876
  Average Sequence Similarity: 0.9234

Phoneme Error Rate (PER):
  Original Model PER: 0.1012 (1234/12189)
  Quantized Model PER: 0.1025 (1249/12189)
  PER Increase (Drop): 0.0013 (+1.28%)
  Average Per-Trial PER Drop: 0.0012

Accuracy:
  Original Model Accuracy: 0.8988 (89.88%)
  Quantized Model Accuracy: 0.8975 (89.75%)
  Accuracy Drop: 0.0013 (+0.14%)

Inference Speed:
  Original Model: 45.23s (0.037 sec/trial)
  Quantized Model: 38.12s (0.031 sec/trial)
  Speedup: 1.19x
```

### CSV Output

The script saves detailed per-trial results to a CSV file with columns:
- `session`, `block`, `trial`: Trial identifiers
- `logit_mse`, `logit_mae`, `logit_cosine_sim`, `logit_max_diff`: Logit comparison metrics
- `seq_edit_distance`, `seq_similarity`: Sequence-level comparison
- `orig_pred_len`, `quant_pred_len`: Prediction lengths
- `orig_pred`, `quant_pred`: Predicted phoneme sequences
- `true_phonemes`: Ground truth phonemes (if `eval_type=val`)
- `orig_per`, `quant_per`, `per_drop`: Per-trial PER metrics (if `eval_type=val`)

## Example Workflow

1. **Quantize models** using `run_all_quantization_configs.sh`:
   ```bash
   bash bci_class/scripts/run_all_quantization_configs.sh \
       --model_path ./data/t15_pretrained_rnn_baseline \
       --gpu_number 0 \
       --skip_lm
   ```

2. **Compare all quantized models**:
   ```bash
   bash bci_class/scripts/compare_all_quantized_models.sh \
       --original_model_path ./data/t15_pretrained_rnn_baseline \
       --quantized_base_dir ./data \
       --gpu_number 0 \
       --eval_type val
   ```

3. **Review results** in `./data/comparison_results/comparison_results_YYYYMMDD_HHMMSS/`

## Notes

- The comparison script does **not** require Redis or the language model
- All comparisons are done at the phoneme level, which is sufficient for measuring quantization impact
- For validation set (`eval_type=val`), PER and accuracy metrics are calculated
- For test set (`eval_type=test`), only logit and sequence similarity metrics are available
- Use `--num_samples` to limit the number of trials for faster testing during development

