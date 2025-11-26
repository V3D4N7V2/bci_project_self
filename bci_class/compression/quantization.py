"""Model quantization for compression.

Supports dynamic and static quantization using PyTorch's quantization APIs.
"""
import torch
import torch.quantization
from typing import Optional, Dict, Any, Tuple
import os
import numpy as np
from torch.utils.data import DataLoader

from model_training import rnn_model
from model_training.dataset import BrainToTextDataset
from model_training.evaluate_model_helpers import load_h5py_file
import pandas as pd


###############################################################################
# Quantization functions


@torch.no_grad()
def quantize_model_dynamic(
    model: 'rnn_model.GRUDecoder',
    dtype: torch.dtype = torch.qint8,
) -> torch.nn.Module:
    """Apply dynamic quantization to the model (Method 1: Dynamic INT8 or FP16).
    
    Dynamic quantization quantizes weights but computes activations in floating point.
    This is suitable for RNNs and works well without calibration data.
    
    Args:
        model: The model to quantize
        dtype: Quantization dtype (torch.qint8 or torch.float16)
    
    Returns:
        Quantized model
    """
    # Dynamic quantization for RNNs requires the model to be on CPU
    # Save original device and move to CPU for quantization
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    
    # For RNNs, we typically quantize the GRU layers and linear layers
    # Dynamic quantization works well for RNNs
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.GRU, torch.nn.Linear},  # Modules to quantize
        dtype=dtype
    )
    
    # Move quantized model back to original device
    quantized_model = quantized_model.to(original_device)
    return quantized_model


@torch.no_grad()
def quantize_model_static_per_channel(
    model: 'rnn_model.GRUDecoder',
    calibration_data_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.qint8,
) -> torch.nn.Module:
    """Apply static quantization with per-channel quantization (Method 3: Static Per-Channel).
    
    Per-channel quantization uses different scale/zero_point for each channel,
    providing better accuracy than per-tensor quantization.
    
    Args:
        model: The model to quantize (must be in eval mode)
        calibration_data_loader: DataLoader with calibration data
        device: Device to run calibration on
        dtype: Quantization dtype (torch.qint8)
    
    Returns:
        Quantized model
    """
    model.eval()
    model.to(device)
    
    # Use per-channel quantization config for better accuracy
    backend = 'fbgemm' if device.type == 'cpu' else 'qnnpack'
    
    # Create per-channel quantization config
    # For per-channel, we use per-channel weight observers
    per_channel_qconfig = torch.quantization.get_default_qconfig(backend)
    # Override weight observer to use per-channel
    if backend == 'fbgemm':
        per_channel_qconfig = torch.quantization.QConfig(
            activation=per_channel_qconfig.activation,
            weight=torch.quantization.default_per_channel_weight_observer
        )
    else:
        # qnnpack may not support per-channel, use default
        per_channel_qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Apply per-channel config to linear layers
    model.qconfig = per_channel_qconfig
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.qconfig = per_channel_qconfig
    
    # Prepare model
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    print("Calibrating model for per-channel static quantization...")
    calibration_batches = 0
    max_calibration_batches = 100
    
    with torch.no_grad():
        for batch in calibration_data_loader:
            if calibration_batches >= max_calibration_batches:
                break
            
            # Extract features and day indices from batch
            # The model forward expects: x (tensor) and day_idx (tensor)
            x = batch['input_features'].to(device)
            day_idx = batch['day_indicies'].to(device)
            
            # Ensure day_idx is a 1D tensor (not a tuple or list)
            if isinstance(day_idx, (list, tuple)):
                day_idx = torch.tensor(day_idx, device=device)
            elif not isinstance(day_idx, torch.Tensor):
                day_idx = torch.tensor([day_idx], device=device)
            
            try:
                _ = model(x=x, day_idx=day_idx, states=None, return_state=False)
                calibration_batches += 1
            except Exception as e:
                print(f"Warning: Error during calibration batch {calibration_batches}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"Calibrated with {calibration_batches} batches")
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # Ensure quantized model is on CPU (static quantized models cannot be moved to GPU)
    quantized_model = quantized_model.cpu()
    
    return quantized_model


@torch.no_grad()
def quantize_model_static(
    model: 'rnn_model.GRUDecoder',
    calibration_data_loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.qint8,
) -> torch.nn.Module:
    """Apply static quantization to the model (Method 2: Static Per-Tensor).
    
    Static quantization quantizes both weights and activations, requiring
    calibration data to determine quantization parameters.
    Uses per-tensor quantization (same scale/zero_point for entire tensor).
    
    Note: Static quantization for RNNs is complex and may not work well.
    Consider using dynamic quantization instead.
    
    Args:
        model: The model to quantize (must be in eval mode)
        calibration_data_loader: DataLoader with calibration data
        device: Device to run calibration on
        dtype: Quantization dtype (torch.qint8)
    
    Returns:
        Quantized model
    """
    model.eval()
    model.to(device)
    
    # For RNNs, static quantization is more complex
    # We'll use a simpler approach: quantize only linear layers statically
    # and use dynamic quantization for GRU layers
    
    # Prepare model for quantization
    # Use fbgemm backend for x86 CPUs, qnnpack for ARM
    backend = 'fbgemm' if device.type == 'cpu' else 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # For RNNs, we typically only quantize linear layers statically
    # GRU layers work better with dynamic quantization
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate with calibration data
    print("Calibrating model for static quantization (per-tensor)...")
    calibration_batches = 0
    max_calibration_batches = 100
    
    with torch.no_grad():
        for batch in calibration_data_loader:
            if calibration_batches >= max_calibration_batches:
                break
            
            # Extract features and day indices from batch
            # The model forward expects: x (tensor) and day_idx (tensor)
            x = batch['input_features'].to(device)
            day_idx = batch['day_indicies'].to(device)
            
            # Ensure day_idx is a 1D tensor (not a tuple or list)
            if isinstance(day_idx, (list, tuple)):
                day_idx = torch.tensor(day_idx, device=device)
            elif not isinstance(day_idx, torch.Tensor):
                day_idx = torch.tensor([day_idx], device=device)
            
            # Run forward pass for calibration
            try:
                _ = model(x=x, day_idx=day_idx, states=None, return_state=False)
                calibration_batches += 1
            except Exception as e:
                print(f"Warning: Error during calibration batch {calibration_batches}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"Calibrated with {calibration_batches} batches")
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # Ensure quantized model is on CPU (static quantized models cannot be moved to GPU)
    quantized_model = quantized_model.cpu()
    
    return quantized_model


def prepare_model_for_qat(
    model: 'rnn_model.GRUDecoder',
    device: torch.device,
) -> torch.nn.Module:
    """Prepare model for Quantization-Aware Training (Method 4: QAT).
    
    QAT simulates quantization during training, allowing the model to adapt
    to quantization and typically achieving better accuracy than post-training quantization.
    
    Args:
        model: The model to prepare (should be in train mode)
        device: Device to prepare model on
    
    Returns:
        Model prepared for QAT
    """
    model.train()
    model.to(device)
    
    # Set quantization config for QAT
    backend = 'fbgemm' if device.type == 'cpu' else 'qnnpack'
    qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig
    
    # Prepare model for QAT
    torch.quantization.prepare_qat(model, inplace=True)
    
    return model


@torch.no_grad()
def convert_qat_model(
    model: 'rnn_model.GRUDecoder',
) -> torch.nn.Module:
    """Convert a QAT model to quantized model.
    
    Call this after QAT training is complete to get the final quantized model.
    
    Args:
        model: QAT model to convert
    
    Returns:
        Quantized model
    """
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    return quantized_model


def create_calibration_data_loader(
    dataset_dir: str,
    sessions: list,
    csv_path: str,
    batch_size: int = 32,
    num_samples: int = 1000,
) -> DataLoader:
    """Create a DataLoader for calibration data.
    
    Args:
        dataset_dir: Directory containing the dataset
        sessions: List of session names to use
        csv_path: Path to CSV file with metadata
        batch_size: Batch size for calibration
        num_samples: Number of samples to use for calibration
    
    Returns:
        DataLoader with calibration data
    """
    import math
    
    b2txt_csv_df = pd.read_csv(csv_path)
    
    # Load a subset of training data for calibration
    # Build trial_indicies dict in the format expected by BrainToTextDataset
    trial_indicies = {}
    samples_per_session = max(1, num_samples // len(sessions))
    total_trials = 0
    
    for session_idx, session in enumerate(sessions):
        train_file = os.path.join(dataset_dir, session, 'data_train.hdf5')
        if os.path.exists(train_file):
            data = load_h5py_file(train_file, b2txt_csv_df)
            num_trials = min(len(data['neural_features']), samples_per_session)
            trial_indicies[session_idx] = {
                'trials': list(range(num_trials)),
                'day_idx': session_idx,
                'session_path': train_file
            }
            total_trials += num_trials
    
    # Check if we have any data
    if len(trial_indicies) == 0:
        raise ValueError(f"No training data found in dataset_dir: {dataset_dir}. "
                        f"Expected files: {[os.path.join(dataset_dir, s, 'data_train.hdf5') for s in sessions]}")
    
    # Calculate number of batches needed for calibration
    # We want enough batches to cover num_samples, but at least 1 batch
    # Since each batch can have batch_size samples, we need ceil(num_samples / batch_size) batches
    # But we're limited by the actual number of trials available
    n_batches = max(1, min(math.ceil(num_samples / batch_size), math.ceil(total_trials / batch_size)))
    
    # Create dataset
    calibration_dataset = BrainToTextDataset(
        trial_indicies=trial_indicies,
        split='train',
        days_per_batch=1,  # One day per batch for calibration
        n_batches=n_batches,  # Calculate based on num_samples and batch_size
        batch_size=batch_size,
        must_include_days=None,
        random_seed=42,
        feature_subset=None
    )
    
    return DataLoader(
        calibration_dataset,
        batch_size=None,  # Dataset already returns batches
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


def save_quantized_model(
    quantized_model: torch.nn.Module,
    save_path: str,
    model_args: Dict[str, Any],
    quantization_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    original_model_path: Optional[str] = None,
):
    """Save a quantized model.
    
    Args:
        quantized_model: The quantized model to save
        save_path: Path to save the model
        model_args: Model configuration arguments
        quantization_type: Type of quantization used (dynamic, static, static_per_channel)
        dtype: Quantization dtype (torch.qint8, torch.float16)
        original_model_path: Path to the original model before quantization
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert dtype to string for saving
    dtype_str = None
    if dtype is not None:
        if dtype == torch.qint8:
            dtype_str = 'qint8'
        elif dtype == torch.float16:
            dtype_str = 'float16'
        else:
            dtype_str = str(dtype)
    
    # Save quantized model state
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_args': model_args,
        'quantized': True,
        'quantization_type': quantization_type,
        'dtype': dtype_str,
        'original_model_path': original_model_path,
    }, save_path)
    
    print(f"Saved quantized model to {save_path}")


def load_quantized_model(
    model_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a quantized model.
    
    Args:
        model_path: Path to the saved quantized model
        device: Device to load the model on
    
    Returns:
        Tuple of (model, model_args)
    """
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    
    model_args = checkpoint.get('model_args', {})
    
    # Recreate model architecture
    model = rnn_model.GRUDecoder(
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_args

