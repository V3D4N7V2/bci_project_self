import argparse

from omegaconf import OmegaConf

from bci_class.compression import distillation

###############################################################################
parser = argparse.ArgumentParser(description='Distillation')

parser.add_argument('--config_path', type=str, help='Path to config file (such as model_training/rnn_args.yaml).')

args = parser.parse_args()
###############################################################################

args2 = OmegaConf.load(args.config_path)
trainer = distillation.BrainToTextDecoder_Trainer(args2)
metrics = trainer.train()
