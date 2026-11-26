"""Image SR checks that need no weights: tile geometry, windows, RoPE and one tiny block against the reference.

    python -m pytest custom_nodes/ComfyUI-KandinskySR/tests   (from the ComfyUI folder, with its Python)

The reference comparisons import a clone of github.com/kandinskylab/kandinsky-image-sr from dev/ref/ and skip when it
is missing.
"""
import importlib
import math
import sys
from pathlib import Path

import pytest
import torch

import comfy.ops
from ksr_pack.kandinsky_sr.image import model as image_model
from ksr_pack.kandinsky_sr.image import upscale

REF = Path(__file__).resolve().parents[1] / "dev/ref/kandinsky-image-sr/src"


def reference(name):
    if not REF.exists():
        pytest.skip("reference clone not present")
    sys.path.insert(0, str(REF))
    return importlib.import_module(f"kandinsky_image_sr.{name}")


@pytest.mark.parametrize("height,width,factor,grid", [(1024, 1024, 2, (3, 3)), (1024, 1024, 4, (5, 5)), (424, 640, 4, (2, 3)),
                                                      (2048, 2048, 4, (11, 11)), (100, 300, 2, (1, 1)), (100, 300, 4, (1, 2))])
def test_tile_grid(height, width, factor, grid):
    lq_tile, tops, lefts = upscale.tile_grid(height, width, factor)
    assert lq_tile == 1024 // factor and (len(tops), len(lefts)) == grid
    assert tops[-1] + lq_tile == max(height, lq_tile) and lefts[-1] + lq_tile == max(width, lq_tile)


def test_tile_positions_match_reference():
    tiling = reference("tiling")
    for length in (1, 255, 256, 257, 424, 640, 1000, 1024, 2048, 4097):
        for tile, stride in ((256, 192), (512, 384)):
            assert upscale.tile_positions(length, tile, stride) == tiling.tile_positions(length, tile, stride)


def test_windows_match_reference():
    tiling = reference("tiling")
    for edges in ((0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 1, 1), (1, 1, 0, 0)):
        ours = upscale.window(64, *edges, device="cpu")
        theirs = tiling.boundary_aware_window(64, 64, top=edges[0], bottom=edges[1], left=edges[2], right=edges[3], device="cpu")
        assert torch.equal(ours, theirs)


def test_rope_matches_reference():
    from diffusers.models.transformers.transformer_kandinsky import Kandinsky5RoPE3D
    rope = Kandinsky5RoPE3D((32, 48, 48))
    h, w = 8, 6
    theirs = rope((1, 1, h, w), (torch.arange(1), torch.arange(h), torch.arange(w))).flatten(1, 3)
    ours = image_model.rope_freqs((32, 48, 48), (h, w), "cpu")
    assert torch.equal(ours.squeeze(2), theirs.squeeze(-4))


def test_creativity_tables_are_baked():
    baked = (0.07, 0.086, 0.1, 0.118, 0.134, 0.15, 0.17, 0.18, 0.1885, 0.207, 0.21, 0.2255, 0.244, 0.27, 0.278, 0.3, 0.312, 0.346, 0.38)
    for factor, values in upscale.CREATIVITY_TIMESTEPS.items():
        assert len(values) == 9 and all(any(abs(v - b) <= 1e-6 for b in baked) for v in values)
    assert upscale.CREATIVITY_TIMESTEPS[2][4] == 0.15 and upscale.CREATIVITY_TIMESTEPS[4][4] == 0.244


def test_tiny_model_matches_reference_fp32():
    """A 2-block model at small width against the reference transformer class, float32 on the CPU."""
    baked = reference("baked_transformer")
    torch.manual_seed(0)
    cfg = dict(in_visual_dim=16, out_visual_dim=16, time_dim=32, patch_size=(1, 2, 2), model_dim=256, ff_dim=512,
               num_visual_blocks=2, axes_dims=(32, 48, 48))
    ref = baked.KandinskyImageSRBakedTransformer3DModel(**cfg, baked_timesteps=(0.15, 0.244), baked_latent_size=16).eval()
    for p in list(ref.parameters()) + list(ref.buffers()):
        if p.dtype == torch.float32 and p.ndim > 0:
            p.data.normal_(0, 0.05)
    ref.set_attention_backend("native")
    ours = image_model.KandinskyImageSR(**cfg, num_timesteps=2, dtype=torch.float32, operations=comfy.ops.manual_cast)
    missing = ours.load_state_dict(ref.state_dict(), strict=True)
    assert not missing.missing_keys and not missing.unexpected_keys
    x = torch.randn(2, 16, 16, 16)
    with torch.inference_mode():
        theirs = ref(hidden_states=x.permute(0, 2, 3, 1).unsqueeze(1), encoder_hidden_states=None,
                     timestep=torch.full((2,), 244.0), pooled_projections=None,
                     visual_rope_pos=(torch.arange(1), torch.arange(8), torch.arange(8)), text_rope_pos=None, return_dict=False)
        got = ours(x, 1)
    theirs = theirs.squeeze(1).permute(0, 3, 1, 2)
    assert got.shape == theirs.shape
    # the reference rounds q and k to bf16 after RoPE even in float32 (tk.py:312); the port follows the input dtype
    assert (got - theirs).abs().max() / theirs.abs().max() < 2e-2
    assert math.isfinite(got.abs().sum().item())


def test_node_schemas():
    from ksr_pack.kandinsky_sr.image import nodes
    assert [n.define_schema().node_id for n in nodes.NODES] == ["KandinskyImageSRLoader", "KandinskyImageSRUpscale"]
    inputs = {i.id: i for i in nodes.KandinskyImageSRUpscale.define_schema().inputs}
    assert list(inputs) == ["model", "vae", "image", "scale", "creativity", "tile_batch_size", "kernels"]
    assert inputs["creativity"].default == 4 and inputs["scale"].default == "4x"


def test_fast_vae_view_leaves_the_decoder_alone():
    from comfy.ldm.modules.diffusionmodules.model import Decoder
    from ksr_pack.kandinsky_sr.image import vae_fast
    decoder = Decoder(ch=32, out_ch=3, ch_mult=[1, 2], num_res_blocks=1, attn_resolutions=[], in_channels=3,
                      resolution=32, z_channels=4)
    flags = {n: (getattr(m, "fused_norm", None), getattr(m, "comfy_cast_weights", None)) for n, m in decoder.named_modules()}
    view = vae_fast._fused_view(decoder)
    assert view is not decoder and view.fused_norm and all(getattr(m, "fused_norm", True) for m in view.modules())
    assert flags == {n: (getattr(m, "fused_norm", None), getattr(m, "comfy_cast_weights", None)) for n, m in decoder.named_modules()}
    assert all(a is b for a, b in zip(view.parameters(), decoder.parameters()))
