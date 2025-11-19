"""Prints the fraction of parameters that are zeros."""
import argparse

import torch

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with magnitude_pruning.')

parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint.')

args = parser.parse_args()
###############################################################################

with torch.no_grad():

    total = 0
    total_zero = 0

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    for n, p in checkpoint['model_state_dict'].items():
        if not n.endswith('._mask'):
            continue
        total += p.numel()
        total_zero += int((p == 0.0).type(torch.int64).sum().detach().cpu().numpy())

    print(f'total: {total}')
    print(f'total_zero: {total_zero}')
    print(f'sparsity fraction: {total_zero / total}')
