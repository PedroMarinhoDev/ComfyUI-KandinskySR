"""Block-sparse attention and fused kernels: index lists on the CPU; on a GPU (KSR_TEST_GPU=1) the NABLA kernel against the
plain-torch block attention and the fused elementwise kernels against the unfused ops, on small shapes."""
import os

import pytest
import torch
import torch.nn.functional as F

from ksr_pack.kandinsky_sr.kernels import fused
from ksr_pack.kandinsky_sr.kernels import nabla as knabla
from ksr_pack.kandinsky_sr.video import nabla

gpu = pytest.mark.skipif(os.environ.get("KSR_TEST_GPU") != "1" or not torch.cuda.is_available(), reason="needs KSR_TEST_GPU=1")


def random_mask(b, h, n, density, generator):
    mask = torch.rand(b, h, n, n, generator=generator) < density
    mask |= torch.eye(n, dtype=torch.bool)  # every query block keeps at least one key block, as the NABLA window does
    return mask


def test_kv_lists():
    g = torch.Generator().manual_seed(0)
    mask = random_mask(2, 3, 9, 0.3, g)
    indices, counts = knabla.kv_lists(mask)
    assert torch.equal(counts, mask.sum(-1, dtype=torch.int32))
    for b in range(2):
        for h in range(3):
            for r in range(9):
                kept = indices[b, h, r, :counts[b, h, r]].tolist()
                assert kept == torch.nonzero(mask[b, h, r]).flatten().tolist()
                assert (indices[b, h, r, counts[b, h, r]:] == 9).all()


def test_unavailable_on_cpu():
    assert not knabla.available("cpu")


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_kernel_matches_block_attention(dtype):
    g = torch.Generator().manual_seed(1)
    b, h, n, d = 1, 4, 12, 64
    q, k, v = (torch.randn(b, h, n * 64, d, generator=g).to("cuda", dtype) for _ in range(3))
    mask = random_mask(b, h, n, 0.35, g).cuda()
    ours = knabla.sparse_attention(q, k, v, mask)
    theirs = nabla.sparse_attention(q, k, v, mask)
    rel = ((ours.float() - theirs.float()).norm() / theirs.float().norm()).item()
    assert rel < (1e-5 if dtype == torch.float32 else 2e-3)
    fast = knabla.sparse_attention(q, k, v, mask, fast=True)
    assert ((fast.float() - theirs.float()).norm() / theirs.float().norm()).item() < 1e-2


@gpu
def test_fused_match_unfused():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 200, 320, generator=g).cuda()
    shift, scale, gate = (torch.randn(2, 1, 320, generator=g).cuda() for _ in range(3))
    o = torch.randn(2, 200, 320, generator=g).cuda().to(torch.bfloat16)
    ref = (F.layer_norm(x, (320,)) * (scale + 1) + shift).to(torch.bfloat16)
    assert (fused.adaln(x, shift, scale, torch.bfloat16) != ref).float().mean() < 1e-3
    nx, normed = fused.residual_adaln(x, gate, o, shift, scale, torch.bfloat16)
    assert torch.equal(nx, x + gate * o.float())
    assert (normed != (F.layer_norm(nx, (320,)) * (scale + 1) + shift).to(torch.bfloat16)).float().mean() < 1e-3
    q = torch.randn(1, 50, 3, 64, generator=g).cuda().to(torch.bfloat16)
    w = torch.rand(64, generator=g).cuda() + 0.5
    ang = torch.rand(1, 50, 1, 32, generator=g).cuda() * 10
    freqs = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).view(1, 50, 1, 32, 2, 2)
    normed = F.rms_norm(q.float(), (64,), w, 1e-6).to(torch.bfloat16)
    ref = (freqs * normed.float().unflatten(-1, (-1, 1, 2))).sum(-1).flatten(-2).to(torch.bfloat16)
    assert (fused.rms_rope(q, w, freqs, 1e-6) != ref).float().mean() < 1e-3
