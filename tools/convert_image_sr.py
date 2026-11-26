"""Convert the Kandinsky ImageSR 2.0 Diffusers transformer (three shards) into one ComfyUI safetensors file.

    python tools/convert_image_sr.py <bundle dir> <output .safetensors>

Keys keep the checkpoint's names. The two timestep tables (time_embed_table, xattn_bias_table) stay float32 as the
reference keeps and adds them; every other tensor is bf16. The modulation layers are float32 in the release but the
reference runs them under bf16 autocast, so their bf16 copy computes the same numbers. The architecture, the baked
timesteps and the pipeline settings go into the file's metadata. Every tensor is checked against the original after
writing; an existing output is left alone.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP32_KEYS = ("time_embed_table", "xattn_bias_table")


def convert(bundle, output):
    if os.path.exists(output):
        print(f"{output} exists, skipped")
        return
    tdir = os.path.join(bundle, "transformer")
    config = json.load(open(os.path.join(tdir, "config.json")))
    pipeline = json.load(open(os.path.join(bundle, "model_index.json")))
    vae = json.load(open(os.path.join(bundle, "vae", "config.json")))
    weight_map = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]

    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(tdir, shard), "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                tensors[key] = t.float() if key.endswith(FP32_KEYS) else t.to(torch.bfloat16)
    assert set(tensors) == set(weight_map), "index and shards disagree"

    arch = {k: config[k] for k in ("in_visual_dim", "out_visual_dim", "time_dim", "patch_size", "model_dim", "ff_dim",
                                   "num_visual_blocks", "axes_dims", "visual_cond", "baked_latent_size")}
    metadata = {
        "model": "kandinsky_image_sr_2.0",
        "architecture": json.dumps(arch),
        "baked_timesteps": json.dumps(config["baked_timesteps"]),
        "default_timesteps": json.dumps(pipeline["one_step_t_by_factor"]),
        "upscale_factors": json.dumps(pipeline["upscale_factors"]),
        "tile_size": str(pipeline["tile_size"]),
        "overlap": str(pipeline["overlap"]),
        "vae_scaling_factor": str(vae["scaling_factor"]),
        "source": "kandinskylab/Kandinsky-ImageSR-2.0 transformer/",
    }
    part = output + ".part"
    save_file(tensors, part, metadata=metadata)
    del tensors
    verify(tdir, weight_map, part)
    os.replace(part, output)
    print(f"wrote {output}")


def verify(tdir, weight_map, path):
    shards = {s: safe_open(os.path.join(tdir, s), "pt") for s in set(weight_map.values())}
    count = 0
    with safe_open(path, "pt") as out:
        assert set(out.keys()) == set(weight_map)
        for key, shard in weight_map.items():
            original, converted = shards[shard].get_tensor(key), out.get_tensor(key)
            expected = original.float() if key.endswith(FP32_KEYS) else original.to(torch.bfloat16)
            assert converted.dtype == expected.dtype and converted.shape == expected.shape, key
            assert torch.equal(converted, expected), key
            if key.endswith(FP32_KEYS):
                assert torch.equal(converted, original), key  # the tables are not rounded
            count += 1
    print(f"verified {count} tensors tensor for tensor")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
