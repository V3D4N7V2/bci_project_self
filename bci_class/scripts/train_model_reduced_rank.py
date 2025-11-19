"""Trains the model with reduced rank.

NOTE: The `init_checkpoint_path` in the config should be set if using this to compress and then train a model.
"""
import argparse

from omegaconf import OmegaConf
import torch

from model_training.rnn_trainer import BrainToTextDecoder_Trainer

from bci_class.compression import svd_compression

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with SVD.')

parser.add_argument('--config_path', type=str, help='Path to config file (such as model_training/rnn_args.yaml).')

parser.add_argument('--day_weights_rank', type=int, help='Rank to compress day_weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank', type=int, help='Rank to compress gru weights to. Leave unset or set to 0 to not compress them.')
parser.add_argument('--gru_rank_ih_l0', type=int, help='Rank to compress weight_ih_l0 to. Only has an effect if gru_rank is not zero/None. If zero/None, equivalent to -gru_rank')

args = parser.parse_args()
###############################################################################

args2 = OmegaConf.load(args.config_path)
trainer = BrainToTextDecoder_Trainer(args2)

# Need to reset these since they will be loaded from the uncompressed model.
trainer.best_val_PER = torch.inf
trainer.best_val_loss = torch.inf


if args.day_weights_rank:
    svd_compression.apply_parameterization_to_day_weights(trainer.model._orig_mod, args.day_weights_rank)

if args.gru_rank:
    svd_compression.apply_parameterization_to_gru(trainer.model._orig_mod, args.gru_rank, args.gru_rank_ih_l0)

metrics = trainer.train()
