"""Unstructured pruning based on magnitudes of weights."""
import collections
from typing import List

import torch

from model_training import rnn_model


###############################################################################


@torch.no_grad()
def prune_day_weights_by_magnitude(model: 'rnn_model.GRUDecoder', retain_fraction: float):
    """Sets day_weights with the lowest (1-retain_fraction) fraction of magnitudes to zero.

    This will do it for each day's weights independently.
    """
    index_to_params = collections.defaultdict(list)

    for n, p in model.named_parameters():
        if not _is_day_weight(n):
            continue
        index = int(n.split('.')[-1])
        index_to_params[index].append(p)

    for parameters in index_to_params.values():
        _prune_parameters(parameters, retain_fraction)


@torch.no_grad()
def prune_non_day_weights_by_magnitude(model: 'rnn_model.GRUDecoder', retain_fraction: float):
    """Sets weights other than the day_weights with the lowest (1-retain_fraction) fraction of magnitudes to zero."""
    parameters = [p for n, p in model.named_parameters() if not _is_day_weight(n)]
    _prune_parameters(parameters, retain_fraction)


def _is_day_weight(n: str) -> bool:
    return n.startswith('day_weights.') or n.startswith('day_biases.')


def _prune_parameters(parameters: List[torch.nn.Parameter], retain_fraction: float):
    """Prunes the parameters in place."""
    v = torch.nn.utils.parameters_to_vector(parameters)

    keep_k = int(retain_fraction * v.numel())
    _, keep_inds = torch.topk(v.abs(), k=keep_k)

    keep_vals = v[keep_inds]

    v.zero_()
    v[keep_inds] = keep_vals

    torch.nn.utils.vector_to_parameters(v, parameters)


###############################################################################
# Parameterization-related stuff.


class Pruned(torch.nn.Module):

    @torch.no_grad()
    def __init__(
        self,
        # Must be the shape shape, dtype, and device as the corresponding weight. Zeros
        # in the mask will be kept zeros. Non-zero values will be set to one.
        #
        # If initializing from a pruned tensor, then you can just use the weight as the mask.
        mask: torch.Tensor,
    ):
        super().__init__()
        self._mask = torch.nn.parameter.Buffer((mask != 0.0).type(mask.dtype))

    def forward(self, X: torch.Tensor):
        return X * self._mask.detach()


def apply_parameterization(
    model: 'rnn_model.GRUDecoder',
):
    """Applies parameterization, using the existing zero values of the weights at the mask."""
    torch.nn.utils.parametrize.register_parametrization(model, 'h0', Pruned(model.h0))

    for i, weight in enumerate(model.day_weights):
        torch.nn.utils.parametrize.register_parametrization(model.day_weights, str(i), Pruned(weight))

    for i, weight in enumerate(model.day_biases):
        torch.nn.utils.parametrize.register_parametrization(model.day_biases, str(i), Pruned(weight))

    for n, p in list(model.gru.named_parameters()):
        torch.nn.utils.parametrize.register_parametrization(model.gru, n, Pruned(p))

    for n, p in list(model.out.named_parameters()):
        torch.nn.utils.parametrize.register_parametrization(model.out, n, Pruned(p))
