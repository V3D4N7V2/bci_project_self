"""Test all 4 quantization methods and compare results."""
import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path


def run_quantization_experiment(
    model_path: str,
    data_dir: str,
    csv_path: str,
    method_name: str,
    quantization_type: str,
    dtype: str,
    gpu_number: int,
    eval_type: str = 'val',
    skip_lm: bool = True,
) -> dict:
    """Run a single quantization experiment.
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*80}")
    print(f"Testing Method: {method_name}")
    print(f"Type: {quantization_type}, Dtype: {dtype}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(model_path),
        f"quantized_{quantization_type}_{dtype}"
    )
    output_model_path = os.path.join(output_dir, 'checkpoint', 'quantized_model.pt')
    
    results = {
        'method': method_name,
        'quantization_type': quantization_type,
        'dtype': dtype,
        'output_dir': output_dir,
        'model_path': output_model_path,
    }
    
    # Step 1: Quantize
    print(f"Step 1: Quantizing model...")
    quantize_start = time.time()
    
    quantize_cmd = [
        sys.executable,
        'bci_class/scripts/quantize_model.py',
        '--model_path', model_path,
        '--output_path', output_model_path,
        '--quantization_type', quantization_type,
        '--dtype', dtype,
        '--data_dir', data_dir,
        '--csv_path', csv_path,
        '--gpu_number', str(gpu_number),
    ]
    
    if quantization_type in ['static', 'static_per_channel']:
        quantize_cmd.extend([
            '--calibration_batch_size', '32',
            '--calibration_samples', '1000',
        ])
    
    result = subprocess.run(
        quantize_cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Quantization failed!")
        print(result.stderr)
        results['quantization_success'] = False
        results['quantization_error'] = result.stderr
        return results
    
    quantization_time = time.time() - quantize_start
    results['quantization_time'] = quantization_time
    results['quantization_success'] = True
    print(f"Quantization completed in {quantization_time:.2f} seconds")
    
    # Step 2: Evaluate
    print(f"\nStep 2: Evaluating quantized model...")
    eval_start = time.time()
    
    eval_cmd = [
        sys.executable,
        'bci_class/scripts/evaluate_quantized_model.py',
        '--model_path', output_model_path,
        '--data_dir', data_dir,
        '--eval_type', eval_type,
        '--csv_path', csv_path,
        '--gpu_number', str(gpu_number),
    ]
    
    if skip_lm:
        eval_cmd.append('--skip_lm')
    
    result = subprocess.run(
        eval_cmd,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed!")
        print(result.stderr)
        results['evaluation_success'] = False
        results['evaluation_error'] = result.stderr
        return results
    
    evaluation_time = time.time() - eval_start
    results['evaluation_time'] = evaluation_time
    results['evaluation_success'] = True
    
    # Try to extract PER from output if available
    output_lines = result.stdout.split('\n')
    for line in output_lines:
        if 'Phoneme Error Rate (PER)' in line:
            try:
                # Extract PER value
                per_str = line.split(':')[1].strip().split()[0]
                results['per'] = float(per_str)
            except:
                pass
    
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    if 'per' in results:
        print(f"PER: {results['per']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test all 4 quantization methods and compare results.'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model directory.')
    parser.add_argument('--data_dir', type=str, default='./data/hdf5_data_final',
                        help='Path to the dataset directory.')
    parser.add_argument('--csv_path', type=str, default='./data/t15_copyTaskData_description.csv',
                        help='Path to CSV file with metadata.')
    parser.add_argument('--eval_type', type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluation type.')
    parser.add_argument('--gpu_number', type=int, default=0,
                        help='GPU number to use (-1 for CPU).')
    parser.add_argument('--skip_lm', action='store_true',
                        help='Skip language model inference for faster evaluation.')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save results JSON. If None, saves to model directory.')
    
    args = parser.parse_args()
    
    # Define the 4 quantization methods to test
    methods = [
        {
            'name': 'Method 1: Dynamic INT8',
            'type': 'dynamic',
            'dtype': 'qint8',
        },
        {
            'name': 'Method 2: Dynamic FP16',
            'type': 'dynamic',
            'dtype': 'float16',
        },
        {
            'name': 'Method 3: Static Per-Tensor INT8',
            'type': 'static',
            'dtype': 'qint8',
        },
        {
            'name': 'Method 4: Static Per-Channel INT8',
            'type': 'static_per_channel',
            'dtype': 'qint8',
        },
    ]
    
    print("=" * 80)
    print("Testing All 4 Quantization Methods")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Evaluation type: {args.eval_type}")
    print(f"GPU: {args.gpu_number}")
    print(f"Total methods to test: {len(methods)}")
    print("=" * 80)
    
    all_results = []
    start_time = time.time()
    
    for i, method in enumerate(methods, 1):
        print(f"\n[{i}/{len(methods)}]")
        result = run_quantization_experiment(
            model_path=args.model_path,
            data_dir=args.data_dir,
            csv_path=args.csv_path,
            method_name=method['name'],
            quantization_type=method['type'],
            dtype=method['dtype'],
            gpu_number=args.gpu_number,
            eval_type=args.eval_type,
            skip_lm=args.skip_lm,
        )
        all_results.append(result)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL METHODS")
    print("=" * 80)
    
    for result in all_results:
        print(f"\n{result['method']}:")
        print(f"  Quantization: {'✓' if result.get('quantization_success') else '✗'}")
        print(f"  Evaluation: {'✓' if result.get('evaluation_success') else '✗'}")
        if 'quantization_time' in result:
            print(f"  Quantization time: {result['quantization_time']:.2f}s")
        if 'evaluation_time' in result:
            print(f"  Evaluation time: {result['evaluation_time']:.2f}s")
        if 'per' in result:
            print(f"  PER: {result['per']:.4f}")
        print(f"  Output: {result['output_dir']}")
    
    # Save results
    if args.output_json is None:
        output_json = os.path.join(
            os.path.dirname(args.model_path),
            f'quantization_comparison_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
    else:
        output_json = args.output_json
    
    summary = {
        'total_time': total_time,
        'methods_tested': len(methods),
        'results': all_results,
    }
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All experiments completed in {total_time:.2f} seconds")
    print(f"Results saved to: {output_json}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

