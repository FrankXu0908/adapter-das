# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Dense/ragged routing contract; runs without a GPU or vendor wheel."""
import sys
import types

import pytest
import torch

from hcu_adapter.fa_vendor import _preserve_dense_varlen, _varlen_triton_ok


@pytest.mark.parametrize("keyword", [False, True])
def test_dense_preserves_batches_and_fa2_return(monkeypatch, keyword):
    q = torch.empty(2, 5, 4, 64)
    k = v = torch.empty(2, 3, 2, 64)
    result = (object(), object(), object())
    calls = []

    def dense(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_func=dense))

    @_preserve_dense_varlen
    def ragged(*args, **kwargs):
        pytest.fail("dense input reached ragged adapter")

    options = dict(softmax_scale=0.125, causal=True, window_size=(4, 0),
                   softcap=2.0, return_softmax_lse=True)
    if keyword:
        actual = ragged(q=q, k=k, v=v, cu_seqlens_q=None, cu_seqlens_k=None, **options)
    else:
        actual = ragged(q, k, v, None, None, **options)
    assert actual is result
    args, kwargs = calls.pop()
    assert args[0] is q and args[1] is k and args[2] is v
    assert kwargs == dict(softmax_scale=0.125, causal=True, window_size=(4, 0),
                          softcap=2.0, return_attn_probs=True)


def test_ragged_preserves_arguments():
    q = k = v = torch.empty(7, 2, 64)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    calls = []
    token = object()

    @_preserve_dense_varlen
    def ragged(*args, **kwargs):
        calls.append((args, kwargs))
        return token

    assert ragged(q, k, v, cu, cu, out=token) is token
    args, kwargs = calls.pop()
    assert args[0] is q and args[3] is cu and args[4] is cu
    assert kwargs == {"out": token}


@pytest.mark.parametrize("boundary", ["neither", "q", "k"])
def test_incomplete_ragged_metadata_rejected(boundary):
    q = k = v = torch.empty(7, 2, 64)
    cu = torch.tensor([0, 7], dtype=torch.int32)
    p = dict(q=q, k=k, v=v, cu_seqlens_q=cu if boundary == "q" else None,
             cu_seqlens_k=cu if boundary == "k" else None)
    assert not _varlen_triton_ok(p)
    with pytest.raises(ValueError, match="both cu_seqlens"):
        _preserve_dense_varlen(lambda **kwargs: pytest.fail("invalid ragged admitted"))(**p)
