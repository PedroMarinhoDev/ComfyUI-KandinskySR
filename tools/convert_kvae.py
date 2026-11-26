"""Convert the KVAE of a Kandinsky 6.0 VSR Diffusers bundle to one ComfyUI safetensors file.

    python tools/convert_kvae.py <bundle>/vae [--dtype bf16|fp32] [--out <file>]

Keys are kept as they are; the architecture and the latent scaling factor go into the metadata under
"kandinsky_sr.kvae". The default dtype is bf16, the dtype the reference runs the KVAE in. The file is written as
<out>.part, checked tensor for tensor against the original, then renamed; an existing output is left alone.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
CONFIG_KEYS = ("in_channels", "out_channels", "z_channels", "ch", "ch_mult", "decoder_ch", "decoder_ch_mult",
               "num_res_blocks", "temporal_compress_times", "temporal_compress_start_level", "scaling_factor")
# what kandinsky_sr/video/kvae.py implements; anything else in the config is refused
FIXED = {"vae_type": "video-kvae", "padding_mode": "zeros", "norm_type": "rms_norm", "double_z": True,
         "downsample_version": 2, "fix_pxs": True, "spatial_factor": 16, "temporal_factor": 4, "resolution": 0}


def default_out(dtype):
    models = Path(os.environ.get("COMFYUI_MODELS", Path.home() / "AI/ComfyUI/models"))
    return models / "vae/kandinsky-sr" / f"kandinsky_kvae_3d_2.0_t4s16_{dtype}.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_dir", type=Path)
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or default_out(args.dtype)
    if out.exists():
        print(f"exists, skipped: {out}")
        return
    source = json.loads((args.vae_dir / "config.json").read_text())
    for key, value in FIXED.items():
        if source.get(key) != value:
            raise ValueError(f"unsupported KVAE config: {key}={source.get(key)!r}, expected {value!r}")
    config = {key: source[key] for key in CONFIG_KEYS}
    weights = args.vae_dir / "diffusion_pytorch_model.safetensors"
    dtype = DTYPES[args.dtype]
    with safe_open(str(weights), framework="pt") as f:
        sd = {key: f.get_tensor(key).to(dtype).contiguous() for key in f.keys()}
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    save_file(sd, str(part), metadata={"kandinsky_sr.kvae": json.dumps(config)})
    del sd
    with safe_open(str(weights), framework="pt") as src, safe_open(str(part), framework="pt") as dst:
        assert sorted(src.keys()) == sorted(dst.keys())
        assert json.loads(dst.metadata()["kandinsky_sr.kvae"]) == config
        for key in src.keys():
            a, b = src.get_tensor(key), dst.get_tensor(key)
            assert b.dtype == dtype and a.shape == b.shape and torch.equal(a.to(dtype), b), key
    umask = os.umask(0)
    os.umask(umask)
    part.chmod(0o666 & ~umask)  # the .part file is created 0600
    part.rename(out)
    print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GB, {args.dtype}), verified tensor for tensor")


if __name__ == "__main__":
    main()
