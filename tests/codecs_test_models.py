"""CPU checks of the codecs models: converted files match the model trees, and the causal segment caches are exact.

    cd ~/AI/ComfyUI && .venv/bin/python -m pytest ~/Projects/ComfyUI-KandinskySR/tests/codecs_test_models.py
"""
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ksr_pack.kandinsky_sr.video import kvae as K
from ksr_pack.kandinsky_sr.video import latent_upscaler as L

MODELS = Path(os.environ.get("COMFYUI_MODELS", Path.home() / "AI/ComfyUI/models"))
KVAE_FILE = MODELS / "vae/kandinsky-sr/kandinsky_kvae_3d_2.0_t4s16_bf16.safetensors"
LU_FILES = [MODELS / f"latent_upscale_models/kandinsky-sr/kandinsky6_sr_latent_upscaler_x{s}_bf16.safetensors" for s in (2, 4)]

TINY = {"in_channels": 3, "out_channels": 3, "z_channels": 8, "ch": 32, "ch_mult": [0.25, 1, 1, 2, 2], "decoder_ch": 32,
        "decoder_ch_mult": [0.25, 1, 1, 2, 2], "num_res_blocks": 1, "temporal_compress_times": 4,
        "temporal_compress_start_level": 1, "scaling_factor": 1.0}


def randomize(model):
    g = torch.Generator().manual_seed(0)
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=p.dtype) * (0.2 if p.ndim > 1 else 1.0)
    return model


def assert_tree_matches(model, path, meta_key):
    with safe_open(str(path), "pt") as f:
        sd = model.state_dict()
        assert set(f.keys()) == set(sd)
        for key in f.keys():
            assert list(f.get_slice(key).get_shape()) == list(sd[key].shape), key
        return json.loads(f.metadata()[meta_key])


@pytest.mark.skipif(not KVAE_FILE.exists(), reason="converted KVAE not present")
def test_kvae_file_matches_tree():
    with safe_open(str(KVAE_FILE), "pt") as f:
        config = json.loads(f.metadata()["kandinsky_sr.kvae"])
    assert_tree_matches(K.KVAEModel(config, dtype=torch.bfloat16, device="meta"), KVAE_FILE, "kandinsky_sr.kvae")


@pytest.mark.parametrize("path", LU_FILES)
def test_upscaler_file_matches_tree(path):
    if not path.exists():
        pytest.skip("converted latent upscaler not present")
    with safe_open(str(path), "pt") as f:
        config = json.loads(f.metadata()["kandinsky_sr.latent_upscaler"])
    assert_tree_matches(L.LatentUpscalerModel(config, dtype=torch.bfloat16, device="meta"), path, "kandinsky_sr.latent_upscaler")


def test_decode_segments_are_exact():
    model = randomize(K.KVAEModel(TINY, dtype=torch.float64, device="cpu"))
    z = torch.randn(1, 8, 9, 3, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        whole = torch.cat(list(model.decode_segments(z, seg=8)), dim=2)
        for seg in (1, 2, 3, 4):
            parts = torch.cat(list(model.decode_segments(z, seg=seg)), dim=2)
            assert parts.shape == whole.shape == (1, 3, 33, 48, 64)
            torch.testing.assert_close(parts, whole, rtol=1e-8, atol=1e-8)  # float64 rounding; a cache bug is O(1)


def test_encode_segments_are_exact():
    model = randomize(K.KVAEModel(TINY, dtype=torch.float64, device="cpu"))
    x = torch.rand(1, 3, 25, 32, 48, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 2 - 1
    def encode(seg_len):
        return torch.cat(list(model.encode_segments(torch.split(x, K.encode_sizes(25, seg_len), dim=2))), dim=2)
    with torch.inference_mode():
        whole = encode(24)
        for seg_len in (4, 8, 16):
            parts = encode(seg_len)
            assert parts.shape == whole.shape == (1, 8, 7, 2, 3)
            torch.testing.assert_close(parts, whole, rtol=1e-8, atol=1e-8)  # float64 rounding; a cache bug is O(1)


def test_encode_sizes_match_the_reference():
    assert K.encode_sizes(121) == [17, 16, 16, 16, 16, 16, 16, 8]
    assert K.encode_sizes(41) == [17, 16, 8]
    assert K.encode_sizes(17) == [17]
    assert K.encode_sizes(1) == [1]


def test_to_uint8_truncates_like_the_reference():
    x = torch.tensor([-1.5, -1.0, 0.0, 0.99, 1.0, 2.0])
    ref = ((((x + 1) * 128) / 255.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)  # vae_io.py:111,128
    assert torch.equal(K.to_uint8(x), ref)


@pytest.mark.parametrize("first", [True, False])
def test_fused_norm_matches_the_module(first):
    from ksr_pack.kandinsky_sr.kernels.conv_norm import spatial_norm_silu
    norm = randomize(K.SpatialNorm3D(16, 8, dtype=torch.float64, device="cpu"))
    zq = torch.randn(1, 8, 3, 4, 6, dtype=torch.float64)
    t = 9 if first else 12  # 1 + 4 * 2 frames on the first segment, 4 * 3 after
    f = torch.randn(1, 16, t, 32, 48, dtype=torch.float64)
    with torch.inference_mode():
        ref = F.silu(norm(f, zq, first, {}))
        out = torch.empty_like(f)
        spatial_norm_silu(f, norm.norm_layer.gamma.view(-1), norm.conv_y(zq), norm.conv_b(zq), first, out)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)  # the fused norm computes in float32
        piece = torch.empty_like(f[..., 8:24, 12:40])  # a spatial piece gathers with the frame's coordinates
        spatial_norm_silu(f[..., 8:24, 12:40], norm.norm_layer.gamma.view(-1), norm.conv_y(zq), norm.conv_b(zq), first, piece, (8, 12, 32, 48))
        torch.testing.assert_close(piece, ref[..., 8:24, 12:40], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("h, w, free_gb", [(64, 96, 22), (128, 192, 22), (135, 240, 22), (128, 192, 12), (32, 48, 22)])
def test_canvas_plan_covers_the_canvas(h, w, free_gb):
    axis, bands = K.canvas_bands(h, w, free_gb * 2**30)
    length = h if axis == 0 else w
    assert axis == (0 if h >= w else 1) and bands[0][0] == 0 and bands[-1][1] == length
    for (s0, e0), (s1, e1) in zip(bands, bands[1:]):
        assert s0 < s1 < e0 < e1 and e0 - s1 >= K.BAND_OVERLAP
    for s, e in bands:
        size = (e - s, w) if axis == 0 else (h, e - s)
        pieces = K.canvas_pieces(4 * size[0], 4 * size[1])
        if pieces is None:
            assert 4 * size[0] <= K.PIECE[0] and 4 * size[1] <= K.PIECE[1]
            continue
        cover = torch.zeros(4 * size[0], 4 * size[1], dtype=torch.int32)
        for y0, y1, x0, x1 in pieces:
            assert y1 - y0 <= K.PIECE[0] and x1 - x0 <= K.PIECE[1]
            cover[y0:y1, x0:x1] += 1
        assert bool((cover == 1).all())
