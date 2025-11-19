"""Unstructured pruning based on magnitudes of weights."""
import os

import argparse
import torch

from bci_class import rnn_model_utils
from bci_class.compression import magnitude_pruning

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with magnitude_pruning.')

parser.add_argument('--input_path', type=str, help='Path to the input pretrained model directory.')
parser.add_argument('--output_path', type=str, help='Path to the output pretrained model directory.')

parser.add_argument('--day_weights_retain_fraction', type=float, help='Fraction of weights to retain from the day_weights. Set to 1.0 to not prune them.')
parser.add_argument('--non_day_weights_retain_fraction', type=float, help='Fraction of weights to retain from the non-day_weights. Set to 1.0 to not prune them.')

args = parser.parse_args()
###############################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    model, model_args = rnn_model_utils.load_model(args.input_path, device)
    print('Model loaded.')

    if args.day_weights_retain_fraction < 1.0:
        magnitude_pruning.prune_day_weights_by_magnitude(model, args.day_weights_retain_fraction)

    if args.non_day_weights_retain_fraction < 1.0:
        magnitude_pruning.prune_non_day_weights_by_magnitude(model, args.non_day_weights_retain_fraction)

    print('Saving model...')
    rnn_model_utils.save_model(model, model_args, args.output_path)
