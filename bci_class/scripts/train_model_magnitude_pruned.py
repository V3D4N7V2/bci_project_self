"""Trains the model with reduced rank.

NOTE: The `init_checkpoint_path` in the config should be set if using this to compress and then train a model.
"""

import argparse

from omegaconf import OmegaConf
import torch

from model_training.rnn_trainer import BrainToTextDecoder_Trainer

from bci_class.compression import magnitude_pruning

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with magnitude_pruning.')

parser.add_argument('--config_path', type=str, help='Path to config file (such as model_training/rnn_args.yaml).')

parser.add_argument('--day_weights_retain_fraction', type=float, help='Fraction of weights to retain from the day_weights. Set to 1.0 to not prune them.')
parser.add_argument('--non_day_weights_retain_fraction', type=float, help='Fraction of weights to retain from the non-day_weights. Set to 1.0 to not prune them.')

args = parser.parse_args()
###############################################################################

args2 = OmegaConf.load(args.config_path)
trainer = BrainToTextDecoder_Trainer(args2)

# Need to reset these since they will be loaded from the uncompressed model.
trainer.best_val_PER = torch.inf
trainer.best_val_loss = torch.inf

with torch.no_grad():
    if args.day_weights_retain_fraction < 1.0:
        magnitude_pruning.prune_day_weights_by_magnitude(trainer.model._orig_mod, args.day_weights_retain_fraction)

    if args.non_day_weights_retain_fraction < 1.0:
        magnitude_pruning.prune_non_day_weights_by_magnitude(trainer.model._orig_mod, args.non_day_weights_retain_fraction)

magnitude_pruning.apply_parameterization(trainer.model._orig_mod)

metrics = trainer.train()
