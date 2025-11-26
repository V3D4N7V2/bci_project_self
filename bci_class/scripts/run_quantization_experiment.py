"""Run a complete quantization experiment: quantize model and evaluate it."""
import os
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Run a complete quantization experiment: quantize a model and evaluate it.'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model directory.')
    parser.add_argument('--data_dir', type=str, default='./data/hdf5_data_final',
                        help='Path to the dataset directory.')
    parser.add_argument('--csv_path', type=str, default='./data/t15_copyTaskData_description.csv',
                        help='Path to CSV file with metadata.')
    parser.add_argument('--quantization_type', type=str, default='dynamic',
                        choices=['dynamic', 'static', 'static_per_channel'],
                        help='Type of quantization: dynamic, static (per-tensor), or static_per_channel.')
    parser.add_argument('--dtype', type=str, default='qint8',
                        choices=['qint8', 'float16'],
                        help='Quantization dtype.')
    parser.add_argument('--eval_type', type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluation type.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use (-1 for CPU).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for quantized model. If None, uses model_path/quantized_{type}_{dtype}.')
    parser.add_argument('--skip_lm', action='store_true',
                        help='Skip language model inference.')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(args.model_path),
            f"quantized_{args.quantization_type}_{args.dtype}"
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_model_path = os.path.join(args.output_dir, 'checkpoint', 'quantized_model.pt')
    
    print("=" * 80)
    print("Step 1: Quantizing model")
    print("=" * 80)
    
    # Step 1: Quantize the model
    quantize_cmd = [
        sys.executable,
        'bci_class/scripts/quantize_model.py',
        '--model_path', args.model_path,
        '--output_path', output_model_path,
        '--quantization_type', args.quantization_type,
        '--dtype', args.dtype,
        '--data_dir', args.data_dir,
        '--csv_path', args.csv_path,
        '--gpu_number', str(args.gpu_number),
    ]
    
    if args.quantization_type == 'static':
        quantize_cmd.extend([
            '--calibration_batch_size', '32',
            '--calibration_samples', '1000',
        ])
    
    print(f"Running: {' '.join(quantize_cmd)}")
    result = subprocess.run(quantize_cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    if result.returncode != 0:
        print(f"Error: Quantization failed with return code {result.returncode}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Step 2: Evaluating quantized model")
    print("=" * 80)
    
    # Step 2: Evaluate the quantized model
    eval_cmd = [
        sys.executable,
        'bci_class/scripts/evaluate_quantized_model.py',
        '--model_path', output_model_path,
        '--data_dir', args.data_dir,
        '--eval_type', args.eval_type,
        '--csv_path', args.csv_path,
        '--gpu_number', str(args.gpu_number),
    ]
    
    if args.skip_lm:
        eval_cmd.append('--skip_lm')
    
    print(f"Running: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    if result.returncode != 0:
        print(f"Error: Evaluation failed with return code {result.returncode}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Experiment complete!")
    print(f"Quantized model saved to: {output_model_path}")
    print(f"Results saved in: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

