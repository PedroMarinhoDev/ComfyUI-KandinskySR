"""Shared setup for the pack's tests: ComfyUI on the path (COMFYUI_PATH, default ~/AI/ComfyUI), the CPU as device
(KSR_TEST_GPU=1 keeps the GPU for the kernel tests), and the pack importable as `ksr_pack` without ComfyUI loading it.

    cd <ComfyUI> && python -m pytest <this pack>/tests
"""
import importlib.util
import os
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get("COMFYUI_PATH", str(Path.home() / "AI/ComfyUI")))

from comfy.cli_args import args  # noqa: E402

args.cpu = os.environ.get("KSR_TEST_GPU") != "1"

if "ksr_pack" not in sys.modules:
    spec = importlib.util.spec_from_file_location("ksr_pack", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    sys.modules["ksr_pack"] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules["ksr_pack"])
