# NOTE: I Guess this does not work. Look into the equation for a GRU.


# """Only accurate for the base model, i.e. without any parameterization."""
# import argparse

# from fvcore.nn import FlopCountAnalysis
# import torch 

# from bci_class import rnn_model_utils

# ###############################################################################
# parser = argparse.ArgumentParser(description='Evaluate a pretrained RNN model on the copy task dataset.')

# parser.add_argument('--input_path', type=str, help='Path to the input pretrained model directory.')

# parser.add_argument('--d_input', type=int, default=512, help='Dimension of each input entry.')
# parser.add_argument('--sequence_length', type=int, default=256, help='Sequence length. FLOPS will be slightly more than directly proportional to this.')

# args = parser.parse_args()
# ###############################################################################

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# with torch.no_grad():
#     model, model_args = rnn_model_utils.load_model(args.input_path, device)

#     # Choose the day index of 0 arbitrarily.
#     flops = FlopCountAnalysis(model, (torch.randn(1, args.sequence_length, args.d_input, device=device, dtype=torch.float32), torch.tensor([0], device=device)))

