"""Script to quantize a pretrained RNN model."""
import os
import torch
import argparse
from omegaconf import OmegaConf

from model_training.rnn_model import GRUDecoder
from bci_class.compression.quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    quantize_model_static_per_channel,
    create_calibration_data_loader,
    save_quantized_model,
)


def main():
    parser = argparse.ArgumentParser(description='Quantize a pretrained RNN model.')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model directory.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the quantized model.')
    parser.add_argument('--quantization_type', type=str, default='dynamic',
                        choices=['dynamic', 'static', 'static_per_channel'],
                        help='Type of quantization: dynamic, static (per-tensor), or static_per_channel.')
    parser.add_argument('--dtype', type=str, default='qint8',
                        choices=['qint8', 'float16'],
                        help='Quantization dtype.')
    parser.add_argument('--data_dir', type=str, default='./data/hdf5_data_final',
                        help='Path to the dataset directory (for static quantization calibration).')
    parser.add_argument('--csv_path', type=str, default='./data/t15_copyTaskData_description.csv',
                        help='Path to CSV file with metadata.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use. Set to -1 to use CPU.')
    parser.add_argument('--calibration_batch_size', type=int, default=32,
                        help='Batch size for calibration data (static quantization only).')
    parser.add_argument('--calibration_samples', type=int, default=1000,
                        help='Number of samples to use for calibration (static quantization only).')
    
    args = parser.parse_args()
    
    # Set up device
    if torch.cuda.is_available() and args.gpu_number >= 0:
        if args.gpu_number >= torch.cuda.device_count():
            raise ValueError(f'GPU number {args.gpu_number} is out of range. Available GPUs: {torch.cuda.device_count()}')
        device = torch.device(f'cuda:{args.gpu_number}')
        print(f'Using {device} for quantization.')
    else:
        if args.gpu_number >= 0:
            print(f'GPU number {args.gpu_number} requested but not available.')
        print('Using CPU for quantization.')
        device = torch.device('cpu')
    
    # Load model args
    model_args = OmegaConf.load(os.path.join(args.model_path, 'checkpoint/args.yaml'))
    
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
    
    # Load model weights
    checkpoint = torch.load(
        os.path.join(args.model_path, 'checkpoint/best_checkpoint'),
        weights_only=False,
        map_location=device
    )
    
    # Handle DataParallel keys
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[new_key] = state_dict[key]
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'qint8': torch.qint8,
        'float16': torch.float16,
    }
    dtype = dtype_map[args.dtype]
    
    # Check if quantized model already exists and is valid
    if os.path.exists(args.output_path):
        try:
            checkpoint = torch.load(args.output_path, weights_only=False, map_location='cpu')
            
            # Check if it's a quantized model
            if checkpoint.get('quantized', False):
                existing_qtype = checkpoint.get('quantization_type')
                existing_dtype = checkpoint.get('dtype', 'qint8')
                
                # Check if quantization type and dtype match
                if existing_qtype == args.quantization_type and existing_dtype == args.dtype:
                    print(f"\nQuantized model already exists at: {args.output_path}")
                    print(f"  Type: {existing_qtype}, Dtype: {existing_dtype}")
                    print(f"  Skipping quantization. Using existing model.")
                    
                    # Verify the model can be loaded
                    try:
                        # Try to load the state dict to verify it's valid
                        state_dict = checkpoint.get('model_state_dict', {})
                        if state_dict:
                            print(f"  Model state dict verified ({len(state_dict)} parameters).")
                            return  # Exit early, model already exists and is valid
                        else:
                            print(f"  Warning: Model state dict is empty, re-quantizing...")
                    except Exception as e:
                        print(f"  Warning: Error verifying model: {e}")
                        print(f"  Re-quantizing...")
                else:
                    print(f"\nExisting quantized model found but with different settings:")
                    print(f"  Existing: type={existing_qtype}, dtype={existing_dtype}")
                    print(f"  Requested: type={args.quantization_type}, dtype={args.dtype}")
                    print(f"  Re-quantizing with new settings...")
            else:
                print(f"\nExisting model found but it's not quantized. Re-quantizing...")
        except Exception as e:
            print(f"\nWarning: Error checking existing model: {e}")
            print(f"  Re-quantizing...")
    
    # Quantize model
    if args.quantization_type == 'dynamic':
        print(f"Applying Method 1: Dynamic quantization (dtype: {args.dtype})...")
        quantized_model = quantize_model_dynamic(model, dtype=dtype)
        print("Dynamic quantization complete!")
        
    elif args.quantization_type == 'static':
        print(f"Applying Method 2: Static quantization per-tensor (dtype: {args.dtype})...")
        
        # Create calibration data loader
        print("Creating calibration data loader...")
        calibration_loader = create_calibration_data_loader(
            dataset_dir=args.data_dir,
            sessions=model_args['dataset']['sessions'],
            csv_path=args.csv_path,
            batch_size=args.calibration_batch_size,
            num_samples=args.calibration_samples,
        )
        
        quantized_model = quantize_model_static(
            model,
            calibration_loader,
            device,
            dtype=dtype
        )
        print("Static quantization (per-tensor) complete!")
        
    elif args.quantization_type == 'static_per_channel':
        print(f"Applying Method 3: Static quantization per-channel (dtype: {args.dtype})...")
        
        # Create calibration data loader
        print("Creating calibration data loader...")
        calibration_loader = create_calibration_data_loader(
            dataset_dir=args.data_dir,
            sessions=model_args['dataset']['sessions'],
            csv_path=args.csv_path,
            batch_size=args.calibration_batch_size,
            num_samples=args.calibration_samples,
        )
        
        quantized_model = quantize_model_static_per_channel(
            model,
            calibration_loader,
            device,
            dtype=dtype
        )
        print("Static quantization (per-channel) complete!")
    
    # Calculate model size reduction
    def get_model_size(model):
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
    
    print(f"\nModel size comparison:")
    print(f"  Original: {original_size / (1024**2):.2f} MB")
    print(f"  Quantized: {quantized_size / (1024**2):.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    # Save quantized model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_quantized_model(
        quantized_model,
        args.output_path,
        dict(model_args),
        quantization_type=args.quantization_type,
        dtype=dtype,
        original_model_path=args.model_path
    )
    
    # Also save args.yaml in the output directory
    output_dir = os.path.dirname(args.output_path)
    args_path = os.path.join(output_dir, 'args.yaml')
    OmegaConf.save(config=model_args, f=args_path)
    
    print(f"\nQuantized model saved to: {args.output_path}")
    print(f"Model args saved to: {args_path}")


if __name__ == '__main__':
    main()

