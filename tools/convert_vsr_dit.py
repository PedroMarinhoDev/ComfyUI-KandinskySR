"""Convert the SR DiT of a Kandinsky 6.0 VSR Diffusers bundle into one ComfyUI safetensors file.

    python tools/convert_vsr_dit.py <bundle dir> <output .safetensors>

Keys keep the checkpoint's names and every tensor keeps its stored dtype: the release has the time embedding, the
AdaLN modulation and the q/k norms in float32, which the reference computes in float32 (explicit fp32 autocast
regions), and everything else in bf16. The architecture, the NABLA settings, the SR settings and the sampler settings
of transformer/config.json and scheduler/scheduler_config.json go into the file's metadata, so the file needs no
folder layout or sidecar. Tensors are streamed one at a time into a .part file, then every tensor is checked against
the original and the file renamed; an existing output is left alone.

Takes the 2.0 and 1.0 pi-Flow bundles (scheduler PiflowScheduler) and the flow-matching teacher.
"""
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

WEIGHTS = "diffusion_pytorch_model.safetensors"
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def read_header(path):
    with open(path, "rb") as f:
        return json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))


def settings(bundle):
    config = json.load(open(os.path.join(bundle, "transformer", "config.json")))
    scheduler = json.load(open(os.path.join(bundle, "scheduler", "scheduler_config.json")))
    if config["use_text"] or not config["visual_cond"] or list(config["patch_size"]) != [1, 1, 1]:
        raise ValueError("expected the text-free SR DiT (use_text false, visual_cond true, patch 1x1x1)")
    sr = config["sr_params"]
    size = str(sr["visual_size"][0])
    attention = config["attention_params"][size]
    if attention["type"] != "nabla" or attention.get("method", "topcdf") != "topcdf":
        raise ValueError(f"unsupported attention for visual size {size}: {attention}")
    if scheduler["_class_name"] == "PiflowScheduler":
        sampler = {"type": "piflow", **{k: scheduler[k] for k in ("nfe", "n_grid", "shift", "num_policy_substeps",
                                                                   "final_step_size_scale", "eps")}}
    elif scheduler["_class_name"] == "FlowMatchEulerDiscreteScheduler":
        sampler = {"type": "euler", "n_grid": 1, "shift": sr["scheduler_scale"], "num_steps": 5}
    else:
        raise ValueError(f"unsupported scheduler {scheduler['_class_name']}")
    if config["out_visual_dim"] != config["in_visual_dim"] * sampler["n_grid"]:
        raise ValueError(f"output head {config['out_visual_dim']} is not in_visual_dim x n_grid")
    architecture = {k: config[k] for k in ("in_visual_dim", "time_dim", "model_dim", "ff_dim", "num_visual_blocks", "axes_dims")}
    architecture["n_grid"] = sampler["n_grid"]
    return {
        "model": os.path.basename(os.path.normpath(bundle)).removesuffix("-Diffusers"),
        "architecture": json.dumps(architecture),
        "attention": json.dumps({k: attention[k] for k in ("P", "wT", "wH", "wW", "add_sta")}),
        "rope_scale": json.dumps(sr["scale_factor"][size]),
        "sr": json.dumps({"visual_size": int(size), "lq_noise_scale": sr["lq_noise_scale"],
                          "lq_noise_type": sr["lq_noise_type"], "fps": sr["fps"]}),
        "sampler": json.dumps(sampler),
        "source": f"kandinskylab/{os.path.basename(os.path.normpath(bundle))} transformer/",
    }


def convert(bundle, output):
    if os.path.exists(output):
        print(f"{output} exists, skipped")
        return
    source = os.path.join(bundle, "transformer", WEIGHTS)
    header = read_header(source)
    header.pop("__metadata__", None)
    keys = sorted(header)
    new, offset = {"__metadata__": settings(bundle)}, 0
    for key in keys:
        entry = header[key]
        size = entry["data_offsets"][1] - entry["data_offsets"][0]
        new[key] = {"dtype": entry["dtype"], "shape": entry["shape"], "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(new, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    part = output + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(source, "rb") as src, open(part, "wb") as dst:
        start = 8 + struct.unpack("<Q", src.read(8))[0]
        dst.write(struct.pack("<Q", len(blob)) + blob)
        for key in keys:
            begin, end = header[key]["data_offsets"]
            src.seek(start + begin)
            dst.write(src.read(end - begin))
    verify(source, part)
    os.replace(part, output)
    print(f"wrote {output}")


def verify(source, path):
    with safe_open(source, "pt") as a, safe_open(path, "pt") as b:
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            x, y = a.get_tensor(key), b.get_tensor(key)
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y), key
        print(f"verified {len(a.keys())} tensors tensor for tensor; metadata {sorted(b.metadata())}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
