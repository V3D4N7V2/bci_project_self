"""Evaluate a quantized RNN model on the copy task dataset."""
import os
import torch
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import *
from bci_class.compression.quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    quantize_model_static_per_channel,
    create_calibration_data_loader,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a quantized RNN model on the copy task dataset.')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the quantized model file.')
    parser.add_argument('--data_dir', type=str, default='./data/hdf5_data_final',
                        help='Path to the dataset directory.')
    parser.add_argument('--eval_type', type=str, default='val', choices=['val', 'test'],
                        help='Evaluation type: "val" for validation set, "test" for test set.')
    parser.add_argument('--csv_path', type=str, default='./data/t15_copyTaskData_description.csv',
                        help='Path to the CSV file with metadata.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use. Set to -1 to use CPU.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save results CSV. If None, saves to model directory.')
    parser.add_argument('--skip_lm', action='store_true',
                        help='Skip language model inference (faster evaluation).')
    
    args = parser.parse_args()
    
    # Quantized models must run on CPU to avoid segfaults
    # Even if GPU is requested, we use CPU for quantized models
    print('Note: Quantized models run on CPU for stability.')
    device = torch.device('cpu')
    
    # Load model
    print("Loading quantized model...")
    checkpoint = torch.load(args.model_path, weights_only=False, map_location=device)
    
    model_args = checkpoint.get('model_args', {})
    if isinstance(model_args, dict) and 'model' not in model_args:
        # Try loading from args.yaml if available
        model_dir = os.path.dirname(args.model_path)
        args_yaml_path = os.path.join(model_dir, 'args.yaml')
        if os.path.exists(args_yaml_path):
            model_args = OmegaConf.load(args_yaml_path)
        else:
            raise ValueError("Could not find model args. Please ensure args.yaml exists in model directory.")
    
    # Get quantization metadata
    quantization_type = checkpoint.get('quantization_type')
    dtype_str = checkpoint.get('dtype', 'qint8')
    original_model_path = checkpoint.get('original_model_path')
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'qint8': torch.qint8,
        'float16': torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.qint8)
    
    # If quantization metadata is missing, try to infer from path
    if quantization_type is None:
        model_path_lower = args.model_path.lower()
        if 'dynamic' in model_path_lower:
            quantization_type = 'dynamic'
        elif 'static_per_channel' in model_path_lower or 'per_channel' in model_path_lower:
            quantization_type = 'static_per_channel'
        elif 'static' in model_path_lower:
            quantization_type = 'static'
        else:
            # Default to dynamic if unknown
            quantization_type = 'dynamic'
            print(f"Warning: Could not determine quantization type, defaulting to 'dynamic'")
    
    if dtype_str is None:
        if 'float16' in args.model_path.lower() or 'fp16' in args.model_path.lower():
            dtype_str = 'float16'
            dtype = torch.float16
        else:
            dtype_str = 'qint8'
            dtype = torch.qint8
    
    print(f"Quantization type: {quantization_type}, dtype: {dtype_str}")
    
    # Load original model first, then quantize it
    if original_model_path and os.path.exists(original_model_path):
        print(f"Loading original model from: {original_model_path}")
        original_checkpoint = torch.load(
            os.path.join(original_model_path, 'checkpoint/best_checkpoint'),
            weights_only=False,
            map_location=device
        )
        
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
        
        # Load original state dict
        original_state_dict = original_checkpoint['model_state_dict']
        new_state_dict = {}
        for key in list(original_state_dict.keys()):
            new_key = key.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = original_state_dict[key]
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        # For static quantization, model must be on CPU
        # For dynamic quantization, we can use the requested device
        if quantization_type in ['static', 'static_per_channel']:
            # Static quantized models MUST stay on CPU
            model = model.cpu()
            calibration_device = torch.device('cpu')
        else:
            model.to(device)
            calibration_device = device
        
        # Quantize the model
        print(f"Quantizing model with {quantization_type} quantization...")
        if quantization_type == 'dynamic':
            model = quantize_model_dynamic(model, dtype=dtype)
        elif quantization_type == 'static':
            # Need calibration data for static quantization
            calibration_loader = create_calibration_data_loader(
                dataset_dir=args.data_dir,
                sessions=model_args['dataset']['sessions'],
                csv_path=args.csv_path,
                batch_size=32,
                num_samples=1000,
            )
            model = quantize_model_static(model, calibration_loader, calibration_device, dtype=dtype)
        elif quantization_type == 'static_per_channel':
            # Need calibration data for static per-channel quantization
            calibration_loader = create_calibration_data_loader(
                dataset_dir=args.data_dir,
                sessions=model_args['dataset']['sessions'],
                csv_path=args.csv_path,
                batch_size=32,
                num_samples=1000,
            )
            model = quantize_model_static_per_channel(model, calibration_loader, calibration_device, dtype=dtype)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Now load the quantized state dict
        quantized_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(quantized_state_dict, strict=False)
        
        # Ensure static quantized models stay on CPU
        if quantization_type in ['static', 'static_per_channel']:
            model = model.cpu()
            device = torch.device('cpu')  # Override device for static quantized models
        
        print("Quantized model loaded successfully!")
    else:
        # Fallback: try to load directly (may not work for quantized models)
        print("Warning: Original model path not found, attempting direct load...")
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
        
        # Try loading with strict=False
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[new_key] = state_dict[key]
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Model loaded (some parameters may be missing due to quantization)")
    
    # Load data
    b2txt_csv_df = pd.read_csv(args.csv_path)
    
    test_data = {}
    total_test_trials = 0
    for session in model_args['dataset']['sessions']:
        files = [f for f in os.listdir(os.path.join(args.data_dir, session)) if f.endswith('.hdf5')]
        if f'data_{args.eval_type}.hdf5' in files:
            eval_file = os.path.join(args.data_dir, session, f'data_{args.eval_type}.hdf5')
            data = load_h5py_file(eval_file, b2txt_csv_df)
            test_data[session] = data
            total_test_trials += len(test_data[session]["neural_features"])
            print(f'Loaded {len(test_data[session]["neural_features"])} {args.eval_type} trials for session {session}.')
    
    print(f'Total number of {args.eval_type} trials: {total_test_trials}')
    print()
    
    # Run inference
    print("Running inference...")
    inference_start_time = time.time()
    
    with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
        for session, data in test_data.items():
            data['logits'] = []
            data['pred_seq'] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in range(len(data['neural_features'])):
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                # Use float32 for CPU inference (bfloat16 may not be supported on CPU)
                # For static quantized models, ensure inputs are on CPU
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
                
                # For static quantized models, ensure model and inputs are on CPU
                if quantization_type in ['static', 'static_per_channel']:
                    neural_input = neural_input.cpu()
                    inference_device = torch.device('cpu')
                else:
                    inference_device = device
                
                logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, inference_device)
                data['logits'].append(logits)
                pbar.update(1)
    
    inference_time = time.time() - inference_start_time
    print(f"Inference completed in {inference_time:.2f} seconds ({inference_time/total_test_trials:.3f} sec/trial)")
    
    # Convert logits to phoneme sequences
    for session, data in test_data.items():
        data['pred_seq'] = []
        for trial in range(len(data['logits'])):
            logits = data['logits'][trial][0]
            pred_seq = np.argmax(logits, axis=-1)
            pred_seq = [int(p) for p in pred_seq if p != 0]
            pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
            pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
            data['pred_seq'].append(pred_seq)
    
    # Calculate PER if validation set
    if args.eval_type == 'val':
        total_edit_distance = 0
        total_seq_length = 0
        
        for session, data in test_data.items():
            for trial in range(len(data['pred_seq'])):
                true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]
                pred_seq = data['pred_seq'][trial]
                
                # Calculate edit distance
                ed = editdistance.eval(true_seq, pred_seq)
                total_edit_distance += ed
                total_seq_length += len(true_seq)
        
        avg_PER = total_edit_distance / total_seq_length if total_seq_length > 0 else 0.0
        print(f"\nPhoneme Error Rate (PER): {avg_PER:.4f} ({total_edit_distance}/{total_seq_length})")
    
    # Language model inference (optional)
    lm_results = {
        'session': [],
        'block': [],
        'trial': [],
        'true_sentence': [],
        'pred_sentence': [],
    }
    
    if not args.skip_lm:
        print("\nRunning language model inference...")
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.flushall()
        
        remote_lm_input_stream = 'remote_lm_input'
        remote_lm_output_partial_stream = 'remote_lm_output_partial'
        remote_lm_output_final_stream = 'remote_lm_output_final'
        
        remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
        remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
        remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)
        
        with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
            for session in test_data.keys():
                for trial in range(len(test_data[session]['logits'])):
                    logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]
                    
                    remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(
                        r, remote_lm_done_resetting_lastEntrySeen
                    )
                    
                    remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                        r,
                        remote_lm_input_stream,
                        remote_lm_output_partial_stream,
                        remote_lm_output_partial_lastEntrySeen,
                        logits,
                    )
                    
                    remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                        r,
                        remote_lm_output_final_stream,
                        remote_lm_output_final_lastEntrySeen,
                    )
                    
                    best_candidate_sentence = lm_out['candidate_sentences'][0]
                    
                    lm_results['session'].append(session)
                    lm_results['block'].append(test_data[session]['block_num'][trial])
                    lm_results['trial'].append(test_data[session]['trial_num'][trial])
                    if args.eval_type == 'val':
                        lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
                    else:
                        lm_results['true_sentence'].append(None)
                    lm_results['pred_sentence'].append(best_candidate_sentence)
                    pbar.update(1)
        
        # Calculate WER if validation set
        if args.eval_type == 'val':
            total_true_length = 0
            total_edit_distance = 0
            
            lm_results['edit_distance'] = []
            lm_results['num_words'] = []
            
            for i in range(len(lm_results['pred_sentence'])):
                true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
                pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
                ed = editdistance.eval(true_sentence.split(), pred_sentence.split())
                
                total_true_length += len(true_sentence.split())
                total_edit_distance += ed
                
                lm_results['edit_distance'].append(ed)
                lm_results['num_words'].append(len(true_sentence.split()))
            
            avg_WER = 100 * total_edit_distance / total_true_length if total_true_length > 0 else 0.0
            print(f"\nWord Error Rate (WER): {avg_WER:.2f}% ({total_edit_distance}/{total_true_length})")
    
    # Save results
    if args.output_csv is None:
        model_dir = os.path.dirname(args.model_path)
        output_file = os.path.join(
            model_dir,
            f'quantized_rnn_{args.eval_type}_results_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        )
    else:
        output_file = args.output_csv
    
    if args.skip_lm:
        # Save phoneme-level results
        results = []
        for session, data in test_data.items():
            for trial in range(len(data['pred_seq'])):
                results.append({
                    'session': session,
                    'block': data['block_num'][trial],
                    'trial': data['trial_num'][trial],
                    'pred_phonemes': ' '.join(data['pred_seq'][trial]),
                })
                if args.eval_type == 'val':
                    true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
                    true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]
                    results[-1]['true_phonemes'] = ' '.join(true_seq)
        
        df_out = pd.DataFrame(results)
    else:
        df_out = pd.DataFrame(lm_results)
    
    df_out.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

