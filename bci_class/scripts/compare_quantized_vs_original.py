"""Compare quantized models against the original model to measure accuracy drop.

This script runs inference on both the original and quantized models and compares:
1. Logit-level differences (MSE, cosine similarity)
2. Phoneme Error Rate (PER) for both models
3. Per-trial detailed comparisons
4. Accuracy drop metrics

No language model is required - all comparisons are at the phoneme level.
"""
import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score

from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import *


def load_model(model_path, device):
    """Load a model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Try loading checkpoint
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        model_args = checkpoint.get('model_args', {})
        is_quantized = checkpoint.get('quantized', False)
        model_dir = os.path.dirname(model_path)
    else:
        # Assume it's a directory with checkpoint/best_checkpoint
        checkpoint_path = os.path.join(model_path, 'checkpoint/best_checkpoint')
        args_yaml_path = os.path.join(model_path, 'checkpoint/args.yaml')
        
        # Check for quantized model first
        quantized_path = os.path.join(model_path, 'checkpoint/quantized_model.pt')
        if os.path.exists(quantized_path):
            checkpoint = torch.load(quantized_path, weights_only=False, map_location=device)
            model_args = checkpoint.get('model_args', {})
            is_quantized = checkpoint.get('quantized', False)
            model_dir = model_path
        elif os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
            model_args = OmegaConf.load(args_yaml_path)
            is_quantized = False
            model_dir = model_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Handle model_args format
    if isinstance(model_args, dict) and 'model' not in model_args:
        args_yaml_path = os.path.join(model_dir, 'args.yaml')
        if os.path.exists(args_yaml_path):
            model_args = OmegaConf.load(args_yaml_path)
        elif os.path.exists(os.path.join(model_dir, 'checkpoint/args.yaml')):
            model_args = OmegaConf.load(os.path.join(model_dir, 'checkpoint/args.yaml'))
    
    # Create model
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'],
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        # If checkpoint is the state dict itself
        state_dict = checkpoint
    
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[new_key] = state_dict[key]
    
    # For quantized models, we may need to handle loading differently
    # Try loading with strict=False first
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}")
        print("Attempting to load with relaxed matching...")
        # Try loading only matching keys
        model_dict = model.state_dict()
        matching_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model.load_state_dict(matching_dict, strict=False)
    
    # For quantized models, they may need to stay on CPU for inference
    # Dynamic quantized RNNs typically work better on CPU
    if is_quantized:
        print("Detected quantized model - keeping on CPU for inference")
        device = torch.device('cpu')
        model = model.cpu()
    else:
        model.to(device)
    
    model.eval()
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Is quantized: {is_quantized}")
    return model, model_args


def run_inference(model, model_args, test_data, device, desc="Running inference"):
    """Run inference on test data and return logits and predictions."""
    logits_dict = {}
    pred_seqs = {}
    
    # Determine actual device (model might be on CPU even if device is CUDA)
    model_device = next(model.parameters()).device
    
    # Disable AMP on CPU (autocast doesn't work well on CPU)
    original_use_amp = model_args.get('use_amp', False)
    if model_device.type == 'cpu':
        model_args = dict(model_args)  # Make a copy to avoid modifying original
        model_args['use_amp'] = False
    
    with tqdm(total=sum(len(data['neural_features']) for data in test_data.values()), 
              desc=desc, unit='trial') as pbar:
        for session, data in test_data.items():
            logits_dict[session] = []
            pred_seqs[session] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                try:
                    neural_input = data['neural_features'][trial]
                    neural_input = np.expand_dims(neural_input, axis=0)
                    # Use model's actual device, not the requested device
                    # Use float32 on CPU, bfloat16 on CUDA
                    if model_device.type == 'cpu':
                        input_dtype = torch.float32
                    else:
                        input_dtype = torch.bfloat16
                    neural_input = torch.tensor(neural_input, device=model_device, dtype=input_dtype)
                    
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, model_device)
                    logits_dict[session].append(logits)
                    
                    # Convert to phoneme sequence
                    # runSingleDecodingStep returns numpy array, not tensor
                    if isinstance(logits, np.ndarray):
                        logits_np = logits
                    else:
                        logits_np = logits[0].cpu().numpy()
                    pred_seq = np.argmax(logits_np, axis=-1)
                    pred_seq = [int(p) for p in pred_seq if p != 0]  # Remove blanks
                    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) 
                               if i == 0 or pred_seq[i] != pred_seq[i-1]]  # Remove duplicates
                    pred_seqs[session].append(pred_seq)
                except Exception as e:
                    print(f"\nError during inference for session {session}, trial {trial}: {e}")
                    # Store None to maintain indexing
                    logits_dict[session].append(None)
                    pred_seqs[session].append([])
                
                pbar.update(1)
    
    return logits_dict, pred_seqs


def compare_logits(logits_orig, logits_quant):
    """Compare logits between original and quantized models."""
    # Handle both numpy arrays and tensors
    if isinstance(logits_orig, torch.Tensor):
        orig_flat = logits_orig.flatten().cpu().numpy()
    else:
        orig_flat = np.array(logits_orig).flatten()
    
    if isinstance(logits_quant, torch.Tensor):
        quant_flat = logits_quant.flatten().cpu().numpy()
    else:
        quant_flat = np.array(logits_quant).flatten()
    
    # MSE
    mse = np.mean((orig_flat - quant_flat) ** 2)
    
    # MAE
    mae = np.mean(np.abs(orig_flat - quant_flat))
    
    # Cosine similarity
    cos_sim = 1 - cosine(orig_flat, quant_flat) if len(orig_flat) > 0 else 0.0
    
    # Max absolute difference
    max_diff = np.max(np.abs(orig_flat - quant_flat))
    
    return {
        'mse': mse,
        'mae': mae,
        'cosine_similarity': cos_sim,
        'max_diff': max_diff
    }


def calculate_per(pred_seqs, true_seqs):
    """Calculate Phoneme Error Rate."""
    total_edit_distance = 0
    total_length = 0
    
    for pred_seq, true_seq in zip(pred_seqs, true_seqs):
        true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_seq]
        pred_phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        
        ed = editdistance.eval(true_phonemes, pred_phonemes)
        total_edit_distance += ed
        total_length += len(true_phonemes)
    
    per = total_edit_distance / total_length if total_length > 0 else 0.0
    return per, total_edit_distance, total_length


def main():
    parser = argparse.ArgumentParser(
        description='Compare quantized models against original model to measure accuracy drop.'
    )
    
    parser.add_argument('--original_model_path', type=str, required=True,
                        help='Path to the original (non-quantized) model directory.')
    parser.add_argument('--quantized_model_path', type=str, required=True,
                        help='Path to the quantized model file or directory.')
    parser.add_argument('--data_dir', type=str, default='./data/hdf5_data_final',
                        help='Path to the dataset directory.')
    parser.add_argument('--eval_type', type=str, default='val', choices=['val', 'test'],
                        help='Evaluation type: "val" for validation set, "test" for test set.')
    parser.add_argument('--csv_path', type=str, default='./data/t15_copyTaskData_description.csv',
                        help='Path to the CSV file with metadata.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use. Set to -1 to use CPU.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save comparison results CSV.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of samples to compare (for faster testing).')
    
    args = parser.parse_args()
    
    # Set up device
    if torch.cuda.is_available() and args.gpu_number >= 0:
        if args.gpu_number >= torch.cuda.device_count():
            raise ValueError(f'GPU number {args.gpu_number} is out of range. '
                           f'Available GPUs: {torch.cuda.device_count()}')
        device = torch.device(f'cuda:{args.gpu_number}')
        print(f'Using {device} for model inference.')
    else:
        if args.gpu_number >= 0:
            print(f'GPU number {args.gpu_number} requested but not available.')
        print('Using CPU for model inference.')
        device = torch.device('cpu')
    
    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    original_model, model_args = load_model(args.original_model_path, device)
    quantized_model, _ = load_model(args.quantized_model_path, device)
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    b2txt_csv_df = pd.read_csv(args.csv_path)
    
    test_data = {}
    total_trials = 0
    for session in model_args['dataset']['sessions']:
        files = [f for f in os.listdir(os.path.join(args.data_dir, session)) 
                if f.endswith('.hdf5')]
        if f'data_{args.eval_type}.hdf5' in files:
            eval_file = os.path.join(args.data_dir, session, f'data_{args.eval_type}.hdf5')
            data = load_h5py_file(eval_file, b2txt_csv_df)
            
            # Limit samples if requested
            if args.num_samples is not None:
                data = {k: v[:args.num_samples] if isinstance(v, list) and len(v) > args.num_samples 
                       else v for k, v in data.items()}
            
            test_data[session] = data
            total_trials += len(test_data[session]["neural_features"])
            print(f'Loaded {len(test_data[session]["neural_features"])} {args.eval_type} '
                  f'trials for session {session}.')
    
    print(f'Total number of {args.eval_type} trials: {total_trials}')
    
    # Run inference on original model
    print("\n" + "="*60)
    print("Running Inference on Original Model")
    print("="*60)
    orig_start = time.time()
    orig_logits, orig_pred_seqs = run_inference(
        original_model, model_args, test_data, device, 
        desc="Original model inference"
    )
    orig_time = time.time() - orig_start
    print(f"Original model inference completed in {orig_time:.2f}s "
          f"({orig_time/total_trials:.3f} sec/trial)")
    
    # Run inference on quantized model
    print("\n" + "="*60)
    print("Running Inference on Quantized Model")
    print("="*60)
    quant_start = time.time()
    quant_logits, quant_pred_seqs = run_inference(
        quantized_model, model_args, test_data, device,
        desc="Quantized model inference"
    )
    quant_time = time.time() - quant_start
    print(f"Quantized model inference completed in {quant_time:.2f}s "
          f"({quant_time/total_trials:.3f} sec/trial)")
    
    # Compare results
    print("\n" + "="*60)
    print("Comparing Results")
    print("="*60)
    
    all_logit_metrics = []
    per_comparisons = []
    detailed_results = []
    
    for session in test_data.keys():
        data = test_data[session]
        
        for trial in range(len(data['neural_features'])):
            # Skip if inference failed (None logits)
            orig_logit = orig_logits[session][trial]
            quant_logit = quant_logits[session][trial]
            
            if orig_logit is None or quant_logit is None:
                print(f"Warning: Skipping comparison for {session}, trial {trial} (inference failed)")
                continue
            
            # Logit comparison
            logit_metrics = compare_logits(orig_logit, quant_logit)
            all_logit_metrics.append(logit_metrics)
            
            # Phoneme sequence comparison
            orig_seq = orig_pred_seqs[session][trial]
            quant_seq = quant_pred_seqs[session][trial]
            
            # Convert to phonemes for comparison
            orig_phonemes = [LOGIT_TO_PHONEME[p] for p in orig_seq]
            quant_phonemes = [LOGIT_TO_PHONEME[p] for p in quant_seq]
            
            # Calculate edit distance between original and quantized predictions
            seq_edit_dist = editdistance.eval(orig_phonemes, quant_phonemes)
            seq_similarity = 1 - (seq_edit_dist / max(len(orig_phonemes), len(quant_phonemes), 1))
            
            # Compare against ground truth if available
            if args.eval_type == 'val':
                true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_seq]
                
                orig_per_trial = editdistance.eval(true_phonemes, orig_phonemes) / len(true_phonemes)
                quant_per_trial = editdistance.eval(true_phonemes, quant_phonemes) / len(true_phonemes)
                per_drop = quant_per_trial - orig_per_trial
                
                per_comparisons.append({
                    'orig_per': orig_per_trial,
                    'quant_per': quant_per_trial,
                    'per_drop': per_drop
                })
            else:
                per_comparisons.append({
                    'orig_per': None,
                    'quant_per': None,
                    'per_drop': None
                })
            
            # Store detailed results
            detailed_results.append({
                'session': session,
                'block': data['block_num'][trial],
                'trial': data['trial_num'][trial],
                'logit_mse': logit_metrics['mse'],
                'logit_mae': logit_metrics['mae'],
                'logit_cosine_sim': logit_metrics['cosine_similarity'],
                'logit_max_diff': logit_metrics['max_diff'],
                'seq_edit_distance': seq_edit_dist,
                'seq_similarity': seq_similarity,
                'orig_pred_len': len(orig_phonemes),
                'quant_pred_len': len(quant_phonemes),
                'orig_pred': ' '.join(orig_phonemes),
                'quant_pred': ' '.join(quant_phonemes),
            })
            
            if args.eval_type == 'val':
                detailed_results[-1].update({
                    'true_phonemes': ' '.join(true_phonemes),
                    'orig_per': per_comparisons[-1]['orig_per'],
                    'quant_per': per_comparisons[-1]['quant_per'],
                    'per_drop': per_comparisons[-1]['per_drop'],
                })
    
    # Calculate aggregate metrics
    print("\n" + "="*60)
    print("Aggregate Metrics")
    print("="*60)
    
    avg_logit_mse = np.mean([m['mse'] for m in all_logit_metrics])
    avg_logit_mae = np.mean([m['mae'] for m in all_logit_metrics])
    avg_logit_cosine = np.mean([m['cosine_similarity'] for m in all_logit_metrics])
    avg_seq_similarity = np.mean([r['seq_similarity'] for r in detailed_results])
    
    print(f"\nLogit-level Comparison:")
    print(f"  Average MSE: {avg_logit_mse:.6f}")
    print(f"  Average MAE: {avg_logit_mae:.6f}")
    print(f"  Average Cosine Similarity: {avg_logit_cosine:.4f}")
    print(f"  Average Sequence Similarity: {avg_seq_similarity:.4f}")
    
    if args.eval_type == 'val':
        # Calculate overall PER for both models
        all_orig_seqs = []
        all_quant_seqs = []
        all_true_seqs = []
        
        for session in test_data.keys():
            data = test_data[session]
            for trial in range(len(data['neural_features'])):
                all_orig_seqs.append(orig_pred_seqs[session][trial])
                all_quant_seqs.append(quant_pred_seqs[session][trial])
                true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                all_true_seqs.append(true_seq)
        
        orig_per, orig_ed, orig_len = calculate_per(all_orig_seqs, all_true_seqs)
        quant_per, quant_ed, quant_len = calculate_per(all_quant_seqs, all_true_seqs)
        per_drop = quant_per - orig_per
        per_drop_pct = (per_drop / orig_per * 100) if orig_per > 0 else 0.0
        
        print(f"\nPhoneme Error Rate (PER):")
        print(f"  Original Model PER: {orig_per:.4f} ({orig_ed}/{orig_len})")
        print(f"  Quantized Model PER: {quant_per:.4f} ({quant_ed}/{quant_len})")
        print(f"  PER Increase (Drop): {per_drop:.4f} ({per_drop_pct:+.2f}%)")
        
        avg_per_drop = np.mean([p['per_drop'] for p in per_comparisons])
        print(f"  Average Per-Trial PER Drop: {avg_per_drop:.4f}")
        
        # Accuracy (1 - PER)
        orig_accuracy = 1 - orig_per
        quant_accuracy = 1 - quant_per
        accuracy_drop = orig_accuracy - quant_accuracy
        accuracy_drop_pct = (accuracy_drop / orig_accuracy * 100) if orig_accuracy > 0 else 0.0
        
        print(f"\nAccuracy:")
        print(f"  Original Model Accuracy: {orig_accuracy:.4f} ({orig_accuracy*100:.2f}%)")
        print(f"  Quantized Model Accuracy: {quant_accuracy:.4f} ({quant_accuracy*100:.2f}%)")
        print(f"  Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop_pct:+.2f}%)")
    
    # Inference speed comparison
    speedup = orig_time / quant_time if quant_time > 0 else 0.0
    print(f"\nInference Speed:")
    print(f"  Original Model: {orig_time:.2f}s ({orig_time/total_trials:.3f} sec/trial)")
    print(f"  Quantized Model: {quant_time:.2f}s ({quant_time/total_trials:.3f} sec/trial)")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Save detailed results
    if args.output_csv is None:
        model_dir = os.path.dirname(args.quantized_model_path)
        if os.path.isdir(args.quantized_model_path):
            model_dir = args.quantized_model_path
        output_file = os.path.join(
            model_dir,
            f'quantization_comparison_{args.eval_type}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        )
    else:
        output_file = args.output_csv
    
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total trials compared: {total_trials}")
    if args.eval_type == 'val':
        print(f"Original PER: {orig_per:.4f}")
        print(f"Quantized PER: {quant_per:.4f}")
        print(f"PER Increase: {per_drop:.4f} ({per_drop_pct:+.2f}%)")
        print(f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop_pct:+.2f}%)")
    print(f"Average Logit Cosine Similarity: {avg_logit_cosine:.4f}")
    print(f"Average Sequence Similarity: {avg_seq_similarity:.4f}")
    print(f"Inference Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    main()

