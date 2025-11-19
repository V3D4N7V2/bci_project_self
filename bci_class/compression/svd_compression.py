"""Compression using simple SVD.

NOTE: Since out.weight.shape = [41, 768], we do not compress it.


"""
from typing import Optional

import torch

from model_training import rnn_model


###############################################################################

@torch.no_grad()
def compress_day_weights_via_svd(
    model: 'rnn_model.GRUDecoder',
    rank: int,
):
    """Compresses the day_weights using SVD.

    NOTE: The weights will be stored as full rank.
    """
    for n, p in model.named_parameters():
        if n.startswith('day_weights.'):
            _compress_parameter(p, rank)


@torch.no_grad()
def compress_gru_via_svd(
    model: 'rnn_model.GRUDecoder',
    rank: int,
    # The gru.weight_ih_l0 has shape [2304, 7168] compared to [2304, 768] for the rest of the parameters. If we
    # want to use a different rank for it, then set this to a a non-None or non-zero value. Having this set to None [default]
    # will result in using 'rank' to compress it.
    rank_ih_l0: Optional[int] = None,
):
    """Compresses the GRU parameters using SVD.

    NOTE: The weights will be stored as full rank.
    """
    for n, p in model.named_parameters():
        if n == 'gru.weight_ih_l0':
            _compress_parameter(p, rank_ih_l0 or rank)
        elif n.startswith('gru.weight_'):
            _compress_parameter(p, rank)


def _compress_parameter(mat: torch.nn.Parameter, rank: int):
    assert len(mat.shape) == 2
    assert rank <= min(*mat.shape)
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    mat[:] = torch.einsum('ir,r,rj->ij', U[:, :rank], S[:rank], Vh[:rank])


###############################################################################
# Parameterization-related stuff.


class ReducedRank(torch.nn.Module):

    def __init__(self, rank: int):
        super().__init__()
        self._rank = rank

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        # NOTE: This looks really weird, but it is how we get this to work with CUDNN. Just returning X @ Y
        # produces the error:
        #   CUDNN_STATUS_NOT_SUPPORTED.This error may appear if you passed in a non-contiguous input
        # We can fix that by setting `torch.backends.cudnn.enabled = False`, but this makes eveything several
        # times slower.
        ret = torch.zeros((X.shape[0], Y.shape[1]), device=X.device, dtype=X.dtype)
        ret[:] = X @ Y
        return ret

    def right_inverse(self, Z):
        assert len(Z.shape) == 2

        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        sqrt_S = torch.sqrt(S[:self._rank])

        return U[:, :self._rank] * sqrt_S, sqrt_S[:, None] * Vh[:self._rank]


def apply_parameterization_to_day_weights(
    model: 'rnn_model.GRUDecoder',
    rank: int,
):
    for i in range(len(model.day_weights)):
        torch.nn.utils.parametrize.register_parametrization(model.day_weights, str(i), ReducedRank(rank))


def apply_parameterization_to_gru(
    model: 'rnn_model.GRUDecoder',
    rank: int,
    # The gru.weight_ih_l0 has shape [2304, 7168] compared to [2304, 768] for the rest of the parameters. If we
    # want to use a different rank for it, then set this to a a non-None or non-zero value. Having this set to None [default]
    # will result in using 'rank' to compress it.
    rank_ih_l0: Optional[int] = None,
):
    for n, p in list(model.gru.named_parameters()):
        if n == 'weight_ih_l0':
            torch.nn.utils.parametrize.register_parametrization(model.gru, n, ReducedRank(rank_ih_l0 or rank))
        elif n.startswith('weight_'):
            torch.nn.utils.parametrize.register_parametrization(model.gru, n, ReducedRank(rank))
