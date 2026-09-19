#!/usr/bin/env python3
"""Export OpenAI's original GPT-2 weights to a TensorFlow-free .npz file.

Why this exists
---------------
``gpt_download.download_and_load_gpt2()`` reads OpenAI's original GPT-2
checkpoint with TensorFlow. TensorFlow and ROCm's PyTorch stack each bundle
their own LLVM and abort the process when both are loaded (see
``setup/01_optional-python-setup-preferences/rocm-setup.md``), so the AMD GPU
environment (``.venv-rocm``) is installed without TensorFlow.

Run this script ONCE in the default environment (``.venv``, which has
TensorFlow) to convert the checkpoint into a plain NumPy archive. The ROCm
environment can then load the exact same weights with ``load_params()``
below, which needs nothing but NumPy.

Usage:
    # in the default (TensorFlow) environment
    uv run python scripts/export_gpt2_params.py --model-size 124M

    # afterwards, in any environment (e.g. the ROCm one), instead of
    # `settings, params = download_and_load_gpt2(...)`:
    #   import sys; sys.path.insert(0, "../../scripts")
    #   from export_gpt2_params import load_params
    #   settings, params = load_params("gpt2/124M_params.npz")
    #   load_weights_into_gpt(gpt, params)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GPT_DOWNLOAD = REPO_ROOT / "ch05" / "01_main-chapter-code" / "gpt_download.py"
MODEL_SIZES = ("124M", "355M", "774M", "1558M")


def _flatten(obj, prefix=""):
    """Flatten the nested params structure into `path -> array` pairs.

    The nesting produced by `gpt_download.py` mixes dicts and lists
    (`params["blocks"][0]["attn"]["c_attn"]["w"]`), so both are encoded into
    a single "/"-separated key, e.g. "blocks/0/attn/c_attn/w".
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten(value, f"{prefix}{key}/")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _flatten(value, f"{prefix}{index}/")
    else:
        yield prefix.rstrip("/"), np.asarray(obj)


def _unflatten(flat):
    """Rebuild the nested dict/list structure produced by `_flatten`."""
    params = {}
    for key, value in flat.items():
        parts = key.split("/")
        cursor = params
        for part, next_part in zip(parts[:-1], parts[1:]):
            child_is_list = next_part.isdigit()
            if part.isdigit():
                index = int(part)
                while len(cursor) <= index:
                    cursor.append([] if child_is_list else {})
                if cursor[index] is None:
                    cursor[index] = [] if child_is_list else {}
                cursor = cursor[index]
            else:
                if part not in cursor:
                    cursor[part] = [] if child_is_list else {}
                cursor = cursor[part]
        last = parts[-1]
        if last.isdigit():
            index = int(last)
            while len(cursor) <= index:
                cursor.append(None)
            cursor[index] = value
        else:
            cursor[last] = value
    return params


def load_params(npz_path):
    """Load `(settings, params)` from an exported .npz file. NumPy only.

    Returns the same structures as
    `gpt_download.download_and_load_gpt2()`, so the result can be passed
    straight to `load_weights_into_gpt(gpt, params)`.
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        settings = json.loads(str(archive["__settings__"]))
        flat = {key: archive[key] for key in archive.files if key != "__settings__"}
    return settings, _unflatten(flat)


def export_params(model_size, models_dir, out_path=None):
    """Download (if needed) and convert a GPT-2 checkpoint to .npz.

    Imports `gpt_download.py` lazily so that this module stays importable
    (for `load_params`) in environments without TensorFlow.
    """
    spec = importlib.util.spec_from_file_location("gpt_download", GPT_DOWNLOAD)
    gpt_download = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpt_download)

    settings, params = gpt_download.download_and_load_gpt2(model_size=model_size, models_dir=str(models_dir))

    if out_path is None:
        out_path = Path(models_dir) / f"{model_size}_params.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat = dict(_flatten(params))
    np.savez(out_path, __settings__=np.array(json.dumps(settings)), **flat)
    return settings, out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", default="124M", choices=MODEL_SIZES)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "gpt2"), help="Where the OpenAI checkpoint is/should be downloaded")
    parser.add_argument("--out", default=None, help="Output .npz path (default: <models-dir>/<model-size>_params.npz)")
    args = parser.parse_args()

    settings, out_path = export_params(args.model_size, args.models_dir, args.out)
    print(f"Settings: {settings}")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("Load it without TensorFlow via:")
    print("    from export_gpt2_params import load_params")
    print(f"    settings, params = load_params({str(out_path)!r})")


if __name__ == "__main__":
    raise SystemExit(main())
