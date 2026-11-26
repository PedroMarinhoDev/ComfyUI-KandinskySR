"""Write the int8 ConvRot variant of the Kandinsky 6.0 VSR 2.0 DiT from the bf16 file of convert_vsr_dit.py: each
block's attention projections (to_query, to_key, to_value, out_layer) and feed-forward linears as int8 with one scale
per output row and a 256-wide block-Hadamard rotation, the format of ComfyUI's int8 ConvRot model files
(comfy-kitchen's TensorWiseINT8Layout, loaded by ComfyUI's mixed-precision ops). The time embedding, modulation, q/k
norms, pooled_bias, input and output layers keep their dtype; the source's metadata is kept.

    python tools/convert_vsr_dit_int8.py <kandinsky6_vsr_2.0_distilled2steps_bf16.safetensors> [out.safetensors]
        [--device cuda:0] [--keep-qk]

--keep-qk leaves to_query and to_key in bf16 (they feed the NABLA block mask). Measured on real tiles it changes
neither the masks nor the output error against bf16 and runs slower, so the default quantizes all six linears.

Prints the relative weight error through the dequantize path for every quantized tensor's sample and the mean.
"""
import argparse
import json
import os
import sys

import torch
from comfy_kitchen.tensor.base import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
from safetensors import safe_open
from safetensors.torch import save_file

INT8 = ("self_attention.to_query", "self_attention.to_key", "self_attention.to_value", "self_attention.out_layer",
        "feed_forward.net.0.proj", "feed_forward.net.2")
QK = ("self_attention.to_query", "self_attention.to_key")
GROUP = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep-qk", action="store_true")
    args = parser.parse_args()
    layers = tuple(x for x in INT8 if not (args.keep_qk and x in QK))
    out = args.out or args.src.replace("_bf16.safetensors", "_int8_convrot.safetensors")
    if os.path.exists(out):
        sys.exit(f"{out} exists")
    marker = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()), dtype=torch.uint8)
    tensors, errors = {}, {}
    with safe_open(args.src, "pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            module = key[:-len(".weight")] if key.endswith(".weight") else None
            if module is not None and module.startswith("visual_transformer_blocks.") and module.split(".", 2)[2] in layers:
                weight = tensor.to(args.device)
                qdata, params = TensorWiseINT8Layout.quantize(weight, per_channel=True, convrot=True, convrot_groupsize=GROUP)
                tensors[key] = qdata.cpu()
                tensors[module + ".weight_scale"] = params.scale.cpu()
                tensors[module + ".comfy_quant"] = marker.clone()
                back = QuantizedTensor(qdata, "TensorWiseINT8Layout", params).dequantize().float()
                errors[module] = ((back - weight.float()).norm() / weight.float().norm()).item()
            else:
                tensors[key] = tensor
    save_file(tensors, out + ".part", metadata=metadata)
    os.replace(out + ".part", out)
    by_kind = {}
    for module, e in errors.items():
        by_kind.setdefault(module.split(".", 2)[2], []).append(e)
    for kind, values in by_kind.items():
        print(f"{kind}: relative weight error mean {sum(values) / len(values):.4f}, max {max(values):.4f}")
    print(f"wrote {out}: {os.path.getsize(out) / 2**30:.2f} GiB, {len(errors)} int8 tensors, "
          f"mean relative weight error {sum(errors.values()) / len(errors):.4f}")


if __name__ == "__main__":
    main()
