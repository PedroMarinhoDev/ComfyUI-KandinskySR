"""Split the latent upscaler bank of a Kandinsky 6.0 VSR Diffusers bundle into an x2 file and an x4 file.

    python tools/convert_latent_upscaler.py <bundle>/latent_upscaler [--dtype bf16|fp32] [--out-dir <dir>]

The bank keeps the 2x model under "_models.0." and the 4x model under "_models.1.". Each file holds only the
tensors its forward reads, renamed to the layout of kandinsky_sr/video/latent_upscaler.py: the x4 model keeps its
own names (its unused 2x supervision head is dropped), the x2 model's private branch is moved onto the same names
and the x4 trunk it carries but never runs is dropped. The architecture goes into the metadata under
"kandinsky_sr.latent_upscaler". The release is bf16, the default output dtype. Each file is written as .part,
checked tensor for tensor against the bank, then renamed; existing outputs are left alone.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
PREFIX = {2: "_models.0.", 4: "_models.1."}
X4_KEEP = ("input_proj.", "pre_blocks.", "upsample_1.", "mid_blocks.", "upsample_2.", "post_blocks.", "output_proj.")
X2_RENAME = {"mid_input_proj.": "input_proj.", "x2_branch.adapter.": "pre_blocks.", "x2_branch.finisher.": "upsample_1.",
             "x2_branch.mid_blocks.": "mid_blocks.", "x2_branch.upsample.": "upsample_2.",
             "x2_branch.blocks.": "post_blocks.", "x2_branch.output_proj.": "output_proj."}
# what kandinsky_sr/video/latent_upscaler.py implements; anything else in the bank config is refused
FIXED = {"architecture": "multi_scale", "bare_stem": True, "bottleneck_channels": None, "depthwise": False, "dims": 3,
         "expand_ratio": 1, "grn": False, "input_skip": False, "kernel_size": 3, "layer_scale_init": None,
         "modulated_norm": True, "modulated_output_proj": True, "temporal_mix": False, "temporal_padding": "replicate",
         "upsample_mode": "pxs_v2", "upsample_padding_mode": "zeros", "upscale_factor": 4}
FIXED_X2 = {"enable_x2_entry": True, "x2_finisher": "pxs_residual", "x2_tail_mode": "private_full"}


def default_out_dir():
    models = Path(os.environ.get("COMFYUI_MODELS", Path.home() / "AI/ComfyUI/models"))
    return models / "latent_upscale_models/kandinsky-sr"


def target_key(scale, key):
    """The output name of a bank key of this scale's model, or None when its forward never reads it."""
    if scale == 4:
        return key if key.startswith(X4_KEEP) else None
    for old, new in X2_RENAME.items():
        if key.startswith(old):
            return new + key[len(old):]
    return None


def convert(bank_dir, scale, dtype_name, out_dir):
    out = out_dir / f"kandinsky6_sr_latent_upscaler_x{scale}_{dtype_name}.safetensors"
    if out.exists():
        print(f"exists, skipped: {out}")
        return
    bank = json.loads((bank_dir / "config.json").read_text())
    model = [m["model"] for m in bank["models"] if m["target_scale"] == f"{scale}x"][0]
    for key, value in {**FIXED, **(FIXED_X2 if scale == 2 else {"enable_x2_entry": False})}.items():
        if model.get(key) != value:
            raise ValueError(f"unsupported latent upscaler config: {key}={model.get(key)!r}, expected {value!r}")
    if model.get("motion_attention") is not None or model.get("global_skip"):
        raise ValueError("unsupported latent upscaler config: motion attention / global skip")
    config = {"scale": scale, "in_channels": model["in_channels"], "stage_channels": model["stage_channels"],
              "num_pre_blocks": model["num_pre_blocks"] if scale == 4 else model["x2_adapter_blocks"],
              "num_mid_blocks": model["num_mid_blocks"], "num_post_blocks": model["num_post_blocks"]}
    dtype = DTYPES[dtype_name]
    weights = bank_dir / "diffusion_pytorch_model.safetensors"
    names = {}
    with safe_open(str(weights), framework="pt") as f:
        for key in f.keys():
            if key.startswith(PREFIX[scale]) and (name := target_key(scale, key[len(PREFIX[scale]):])) is not None:
                names[name] = key
        sd = {name: f.get_tensor(key).to(dtype).contiguous() for name, key in names.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    save_file(sd, str(part), metadata={"kandinsky_sr.latent_upscaler": json.dumps(config)})
    del sd
    with safe_open(str(weights), framework="pt") as src, safe_open(str(part), framework="pt") as dst:
        assert sorted(dst.keys()) == sorted(names)
        assert json.loads(dst.metadata()["kandinsky_sr.latent_upscaler"]) == config
        for name, key in names.items():
            a, b = src.get_tensor(key), dst.get_tensor(name)
            assert b.dtype == dtype and a.shape == b.shape and torch.equal(a.to(dtype), b), name
    umask = os.umask(0)
    os.umask(umask)
    part.chmod(0o666 & ~umask)  # the .part file is created 0600
    part.rename(out)
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GB, {len(names)} tensors, {dtype_name}), verified tensor for tensor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bank_dir", type=Path)
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    for scale in (2, 4):
        convert(args.bank_dir, scale, args.dtype, args.out_dir or default_out_dir())


if __name__ == "__main__":
    main()
