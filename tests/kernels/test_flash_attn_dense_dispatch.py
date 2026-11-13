# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Real SGLang dispatcher: dense cross-attention and ragged regression.

Run in a fresh SGLang-host process for each HCU_FA_IMPL=auto/triton/vendor.
CPU FP64 math is independent of the vendor and adapter implementations.
"""
import pytest
import torch


def _call(q, k, v, cu_q=None, cu_k=None, **kwargs):
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
    return flash_attn_varlen_func(q, k, v, cu_q, cu_k, **kwargs)


def _reference(q, k, v, causal=False):
    q, k, v = (x.detach().cpu().double().transpose(1, 2) for x in (q, k, v))
    repeats = q.shape[1] // k.shape[1]
    k, v = (x.repeat_interleave(repeats, dim=1) for x in (k, v))
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        mask = torch.arange(sk)[None, :] > torch.arange(sq)[:, None] + sk - sq
        scores.masked_fill_(mask, -torch.inf)
    return (scores.softmax(-1) @ v).transpose(1, 2), scores.logsumexp(-1)


def _check(actual, expected):
    assert actual.shape == expected.shape
    actual = actual.detach().cpu().double()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    cos = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert cos >= 0.9999


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lse", [False, True])
@pytest.mark.parametrize("heads", [(2, 2), (4, 2)])
def test_dense_cross_attention(dtype, causal, lse, heads):
    torch.manual_seed(1729)
    hq, hk = heads
    q = torch.randn(2, 3, hq, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 5, hk, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    result = _call(q, k, v, causal=causal, return_softmax_lse=lse)
    expected, expected_lse = _reference(q, k, v, causal)
    _check(result[0] if lse else result, expected)
    if lse:
        # Match upstream FA2's return_attn_probs tuple without repacking it.
        assert len(result) == 3
        _check(result[1], expected_lse)


def test_dense_changed_input_graph():
    q = torch.randn(2, 3, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 5, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for _ in range(3):
        _call(q, k, v)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _call(q, k, v)
    for seed in (11, 12, 13):
        torch.manual_seed(seed)
        q.normal_()
        k.normal_()
        v.normal_()
        graph.replay()
        _check(result, _reference(q, k, v)[0])


def test_ragged_public_dispatch():
    q = torch.randn(7, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(9, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 5, 9], device="cuda", dtype=torch.int32)
    result = _call(q, k, v, cu_q, cu_k, max_seqlen_q=4, max_seqlen_k=5)
    expected = torch.cat([
        _reference(q[:3][None], k[:5][None], v[:5][None])[0][0],
        _reference(q[3:][None], k[5:][None], v[5:][None])[0][0],
    ])
    _check(result, expected)
