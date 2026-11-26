"""Video checks that need no weights: frame plan, padding and tile grid (golden values from the reference's own
functions, 2026-11-26), NABLA token order, window and attention, the pi-Flow rollout against a captured reference
rollout (skipped without dev/captures), the chunked stitch, and a tiny DiT.

    cd ~/AI/ComfyUI && .venv/bin/python -m pytest ~/Projects/ComfyUI-KandinskySR/tests/video_test_core.py
"""
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import comfy.ops

from ksr_pack.kandinsky_sr.video import nabla, pipeline, sampler
from ksr_pack.kandinsky_sr.video.model import KandinskyVSRDiT

CAPTURES = Path(__file__).resolve().parents[1] / "dev/captures/video"


@pytest.mark.parametrize("total,fps,count,first,last,out_fps", [
    (180, 30.0, 144, [0, 1, 2, 4, 5], [176, 178, 179], 24),
    (149, 29.97002997, 119, [0, 1, 2, 4, 5], [145, 146, 147], 24),
    (121, 24.0, 121, [0, 1, 2, 3, 4], [118, 119, 120], 24),
    (500, 60.0, 200, [0, 2, 5, 8, 10], [492, 495, 498], 24),
    (50, 15.0, 50, [0, 1, 2, 3, 4], [47, 48, 49], 15),
    (400, 25.0, 400, [0, 1, 2, 3, 4], [397, 398, 399], 25),
])
def test_frame_plan(total, fps, count, first, last, out_fps):
    indices, rate = pipeline.frame_plan(total, fps)
    assert len(indices) == count and indices[:5] == first and indices[-3:] == last
    assert round(rate) == out_fps


def test_aligned_length():
    assert [pipeline.aligned_length(n) for n in (1, 2, 9, 17, 121, 137, 144, 361)] == [1, 9, 9, 17, 121, 137, 145, 361]


@pytest.mark.parametrize("h,w,scale,padded,tile,tops,lefts", [
    (512, 768, 2, (512, 768), (256, 384), (0, 128, 256), (0, 192, 384)),
    (512, 768, 4, (512, 768), (128, 192), (0, 96, 192, 288, 384), (0, 144, 288, 432, 576)),
    (360, 360, 2, (368, 368), (256, 256), (0, 112), (0, 112)),
    (300, 420, 2, (304, 432), (256, 384), (0, 48), (0, 48)),
    (200, 200, 2, (256, 256), (256, 256), (0,), (0,)),
    (1080, 1920, 2, (1088, 1920), (256, 384), (0, 160, 336, 496, 672, 832), (0, 256, 512, 768, 1024, 1280, 1536)),
    (480, 854, 4, (480, 864), (128, 192), (0, 96, 176, 256, 352), (0, 128, 272, 400, 544, 672)),
    (17, 33, 4, (128, 192), (128, 192), (0,), (0,)),
    (720, 720, 4, (720, 720), (128, 128), (0, 80, 176, 256, 336, 416, 512, 592), (0, 80, 176, 256, 336, 416, 512, 592)),
])
def test_grid(h, w, scale, padded, tile, tops, lefts):
    plan = pipeline.Plan(9, h, w, float(scale), 0.2)
    assert (plan.padded, plan.tile, plan.tops, plan.lefts) == (padded, tile, tops, lefts)
    assert plan.out_size == (h * scale, w * scale)


def test_pre_upscale_route():
    plan = pipeline.Plan(9, 512, 768, 2.25, 0.2)
    assert (plan.height, plan.width, plan.out_size) == (576, 864, (1152, 1728))
    assert pipeline.pre_upscale(torch.zeros(1, 3, 360, 360, dtype=torch.uint8), 1.125).shape[-2:] == (400, 400)


def test_pad_mirrors_then_repeats():
    video = torch.arange(2 * 3 * 5 * 6, dtype=torch.uint8).view(2, 3, 5, 6)
    padded = pipeline.pad(video, 8, 8)
    assert torch.equal(padded[..., :5, :6], video) and torch.equal(padded[..., 5, :6], video[..., 3, :])
    small = video[..., :2, :2]
    assert torch.equal(pipeline.pad(small, 4, 4)[..., 3, 0], small[..., 1, 0])  # strip as long as the source


def test_block_order():
    t, h, w = 2, 16, 24
    grid = nabla.Grid(t, h, w, 0.8, 11, 7, 7)
    raster = torch.arange(t * h * w).view(1, -1, 1)
    blocks = nabla.to_blocks(raster, grid)[0, :, 0]
    expected = torch.arange(t * h * w).view(t, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4).flatten()
    assert torch.equal(blocks, expected)
    assert torch.equal(nabla.from_blocks(nabla.to_blocks(raster, grid), grid), raster)


def test_sta_window():
    grid = nabla.Grid(13, 32, 48, 0.8, 11, 7, 7)
    sta = grid.sta("cpu")
    t, hb, wb = grid.blocks
    coords = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(hb), torch.arange(wb), indexing="ij"), -1).view(-1, 3)
    d = (coords[:, None] - coords[None]).abs()
    assert torch.equal(sta, (d[..., 0] <= 5) & (d[..., 1] <= 3) & (d[..., 2] <= 3))
    assert not sta.all()  # 6 block columns: the +-3 window does not reach across


def test_block_mask_topcdf():
    torch.manual_seed(0)
    grid = nabla.Grid(3, 16, 16, 0.8, 1, 1, 1, add_sta=False)
    q, k = torch.randn(2, 1, 4, 3 * 256, 64).unbind(0)
    mask = nabla.block_mask(q, k, grid)
    probs = torch.softmax(q.view(1, 4, 12, 64, 64).mean(-2) @ k.view(1, 4, 12, 64, 64).mean(-2).transpose(-1, -2) / 8, -1)
    kept = (probs * mask).sum(-1)
    dropped_max = torch.where(mask, 0.0, probs).amax(-1)
    assert (kept >= 0.8 - 1e-6).all()
    assert ((kept - torch.where(mask, probs, 2.0).amin(-1)) < 0.8 + 1e-6).all()  # minimal: dropping the smallest kept fails
    assert (dropped_max <= torch.where(mask, probs, 2.0).amin(-1) + 1e-7).all()  # kept blocks are the largest


def test_sparse_attention_is_masked_attention():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 3, 6 * 64, 32).unbind(0)
    mask = torch.rand(2, 3, 6, 6) < 0.5
    mask |= torch.eye(6, dtype=torch.bool)
    ours = nabla.sparse_attention(q, k, v, mask, budget=1 << 16)
    dense = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.repeat_interleave(64, -1).repeat_interleave(64, -2))
    torch.testing.assert_close(ours, dense, rtol=1e-5, atol=1e-5)


def test_schedule():
    steps = sampler.schedule(2, 0.5, 5.0, 1e-6, "cpu")
    assert [torch.tensor(s[0], dtype=torch.float32).item() for s in steps] == [1000.0, 714.28564453125]  # meta.json
    assert [s[2:] for s in steps] == [pytest.approx((1.0, 1 / 3, 2 / 3)), pytest.approx((1 / 3, 0.0, 1 / 3))]


def _capture():
    for card in ("4090", "3090"):
        path = CAPTURES / card / "small" / "tile_00.pt"
        if path.exists():
            return torch.load(path, weights_only=False)
    pytest.skip("no small-case capture in dev/captures/video")


def test_rollout_matches_reference_capture():
    """The pi-Flow rollout on the reference's recorded DiT outputs gives the reference's states bit for bit."""
    tile = _capture()
    x = tile["init"]
    for step, (timestep, sigma, raw_src, raw_dst, seg) in enumerate(sampler.schedule(2, 0.5, 5.0, 1e-6, "cpu")):
        assert torch.equal(tile["dit_in"][step], x)
        x = sampler.rollout(x, tile["dit_out"][step], sigma, raw_src, raw_dst, seg, 128, 5.0, 1e-6)
        assert torch.equal(x, tile["rollout"][step])
    assert torch.equal(x, tile["final_latent"])


def test_stitch_chunks_equal_whole_clip():
    """Frame-chunked stitch = the reference's whole-clip CPU stitch (tiling_utils.py:198) bit for bit, with the true
    summed window in the corners, where the reference clamps it at 1e-6 and darkens them."""
    torch.manual_seed(0)
    plan = pipeline.Plan(11, 300, 420, 2.0, 0.2)
    tiles = [torch.randint(0, 256, (3, 11, 512, 768), dtype=torch.uint8) for _ in plan.tiles]
    out = []
    pipeline.stitch(tiles, plan, torch.device("cpu"), out.append)
    ours = torch.cat(out)
    window = (torch.hann_window(514)[1:-1, None] * torch.hann_window(770)[None, 1:-1])[None, None]
    acc, weight = torch.zeros(3, 11, 608, 864), torch.zeros(1, 1, 608, 864)
    for (top, left), tile in zip(plan.tiles, tiles):
        acc[:, :, 2 * top:2 * top + 512, 2 * left:2 * left + 768] += tile.float() * window
        weight[:, :, 2 * top:2 * top + 512, 2 * left:2 * left + 768] += window
    ref = (acc / weight).clamp(0, 255).to(torch.uint8)[..., :600, :840].permute(1, 2, 3, 0)
    assert torch.equal(ours, ref)
    clamped = (acc / weight.clamp(min=1e-6)).clamp(0, 255).to(torch.uint8)[..., :600, :840].permute(1, 2, 3, 0)
    corners = (weight[0, 0] < 1e-6)[:600, :840]  # where the reference's clamp acts: i * j < ~40 px from a corner
    assert torch.equal(ours[:, ~corners], clamped[:, ~corners]) and not torch.equal(ours[:, 0, :4], clamped[:, 0, :4])


def test_tiny_dit_nabla_with_full_mask_is_dense():
    torch.manual_seed(0)
    attention = {"P": 1.0, "wT": 1, "wH": 1, "wW": 1, "add_sta": True}
    model = KandinskyVSRDiT(in_visual_dim=4, n_grid=2, time_dim=32, model_dim=128, ff_dim=256, num_visual_blocks=2,
                            axes_dims=(16, 24, 24), attention=attention, dtype=torch.float32, operations=comfy.ops.manual_cast)
    for p in model.parameters():
        p.data = torch.randn_like(p) * 0.05
    x = torch.randn(1, 2, 8, 16, 9)
    t = torch.tensor([1000.0])
    dense_model = KandinskyVSRDiT(in_visual_dim=4, n_grid=2, time_dim=32, model_dim=128, ff_dim=256, num_visual_blocks=2,
                                  axes_dims=(16, 24, 24), dtype=torch.float32, operations=comfy.ops.manual_cast)
    dense_model.load_state_dict(model.state_dict())
    sparse = model(x, t)
    dense = dense_model(x, t)
    assert sparse.shape == (1, 2, 8, 16, 8)
    torch.testing.assert_close(sparse, dense, rtol=1e-4, atol=1e-5)


def test_converted_file_matches_model():
    path = Path.home() / "AI/ComfyUI/models/diffusion_models/kandinsky-sr/kandinsky6_vsr_2.0_distilled2steps_bf16.safetensors"
    if not path.exists():
        pytest.skip("converted DiT not in models/diffusion_models/kandinsky-sr")
    import json
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        meta, keys = f.metadata(), set(f.keys())
        shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
    arch = json.loads(meta["architecture"])
    with torch.device("meta"):
        model = KandinskyVSRDiT(**arch, rope_scale=json.loads(meta["rope_scale"]), attention=json.loads(meta["attention"]),
                                dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    params = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    assert params == shapes
    assert json.loads(meta["sampler"])["nfe"] == 2 and math.isclose(json.loads(meta["sr"])["lq_noise_scale"], 0.7)


def test_node_schemas():
    from ksr_pack.kandinsky_sr.video import nodes
    schemas = {n.define_schema().node_id: n.define_schema() for n in nodes.NODES}
    assert set(schemas) == {"KandinskyVideoSRLoader", "KandinskyVideoSRUpscale"}
    upscale = {i.id: i for i in schemas["KandinskyVideoSRUpscale"].inputs}
    assert upscale["scale"].options == ["2x", "2.25x", "4x"] and upscale["seed"].default == 42
    assert upscale["video"].optional and upscale["images"].optional


def test_blend_latents_keeps_a_constant_field():
    plan = pipeline.Plan(9, 300, 420, 2.0, 0.2)
    tiles = [torch.full((3, 32, 48, 4), 0.75) for _ in plan.tiles]
    canvas = pipeline.blend_latents(tiles, plan, torch.device("cpu"))
    assert canvas.shape == (3, 38, 54, 4)  # the padded 304x432 frame x2 / 16
    torch.testing.assert_close(canvas, torch.full_like(canvas, 0.75))


@pytest.mark.parametrize("count,window,starts,padded", [
    (121, 361, [0], 121),
    (137, 361, [0], 137),
    (144, 361, [0], 145),
    (457, 121, [0, 112, 224, 336], 457),
    (300, 121, [0, 112, 184], 305),
    (1000, 361, [0, 352, 640], 1001),
])
def test_window_starts(count, window, starts, padded):
    got, length = pipeline.window_starts(count, window)
    assert (got, length) == (starts, padded)
    assert all(s % 8 == 0 for s in got) and length >= count
    if len(got) > 1:
        assert all(b - a <= window - pipeline.WINDOW_OVERLAP for a, b in zip(got, got[1:]))  # every join shares 9+


def test_window_noise_matches_on_shared_frames():
    lq = torch.zeros(31, 4, 6, 8)
    a = sampler.degraded_start(lq, 7, 0.7, "ddpm", first_frame=0)
    b = sampler.degraded_start(lq, 7, 0.7, "ddpm", first_frame=28)  # a window 112 frames later
    assert torch.equal(a[28:], b[:3]) and not torch.equal(a[:3], b[:3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")
def test_cpu_noise_is_the_same_on_any_device():
    lq = torch.zeros(5, 8, 12, 64)
    on_cpu = sampler.degraded_start(lq, 42, 0.7, "ddpm")
    on_card = sampler.degraded_start(lq.cuda(), 42, 0.7, "ddpm").cpu()
    assert torch.equal(on_cpu, on_card)



def test_slow_card_leaves_the_last_tiles_to_a_faster_one():
    """9 tiles, one card 3x slower: the greedy queue gives the slow card 3 tiles and ends at 9 units; leaving the last
    tile to the fast card ends at 7 (fast 7 tiles, slow 2)."""
    class Card:
        def __init__(self, seconds):
            self.seconds, self.device, self.tiles = seconds, torch.device("cpu"), 0

    def work(card, index):
        time.sleep(card.seconds)
        card.tiles += 1

    fast, slow = Card(0.05), Card(0.15)
    timings = {}
    pipeline.run_tiles([fast, slow], 9, work, timings, "dit")
    assert (fast.tiles, slow.tiles) == (7, 2) and timings["dit"] < 0.42, (fast.tiles, slow.tiles, timings)


def test_short_file_plans_from_the_frames_it_holds(monkeypatch, tmp_path):
    """A container that says 362 frames for a 361-frame file: one window planned, the progress total of one window,
    every frame written."""
    from ksr_pack.kandinsky_sr.video import nodes

    class Source:
        total, fps, audio = 362, 24.0, None

        def get(self, start, stop):
            return np.zeros((max(0, min(stop, 361) - start), 64, 96, 3), dtype=np.uint8)

    class Card:
        device = torch.device("cpu")
        dit = type("Dit", (), {"sampler": {"nfe": 2}})

    class Models:
        cards = [Card()]

        def upscaler(self, tiling):
            return None

    totals, written, windows = [], [], []

    def upscale(cards, upscaler, video, plan, seed, emit, **kwargs):
        windows.append(video.shape[0])
        emit(torch.zeros(video.shape[0], *plan.out_size, 3, dtype=torch.uint8))
        return {}

    class Writer:
        def __init__(self, *args):
            pass

        def put(self, frames):
            written.append(frames.shape[0])

        def close(self):
            pass

    monkeypatch.setattr(nodes.pipeline, "upscale", upscale)
    monkeypatch.setattr(nodes, "Writer", Writer)
    monkeypatch.setattr(nodes, "pick_encoder", lambda *args: "test")
    monkeypatch.setattr(nodes.folder_paths, "get_temp_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes.comfy.utils, "ProgressBar", lambda total: totals.append(total))
    nodes.KandinskyVideoSRUpscale.run(Models(), Source(), "2x", 42, 0.2, "auto", "fast", "pixels", 361, torch.device("cpu"), 0.0)
    assert windows == [361] and sum(written) == 361
    assert totals == [len(pipeline.Plan(361, 64, 96, 2.0, 0.2).tiles) * 3]
