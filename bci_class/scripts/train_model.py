"""Better version of the train_model script."""
import argparse

from omegaconf import OmegaConf

from model_training.rnn_trainer import BrainToTextDecoder_Trainer

###############################################################################
parser = argparse.ArgumentParser(description='Compresses an RNN model with SVD.')

parser.add_argument('--config_path', type=str, help='Path to config file (such as model_training/rnn_args.yaml).')

args = parser.parse_args()
###############################################################################

args2 = OmegaConf.load(args.config_path)
trainer = BrainToTextDecoder_Trainer(args2)
metrics = trainer.train()
