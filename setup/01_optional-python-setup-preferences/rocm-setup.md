# Using an AMD GPU (ROCm) Instead of NVIDIA/CUDA

This project's default `uv sync` installs a CPU/CUDA-appropriate PyTorch build automatically. If you have an **AMD GPU** on Linux, uv has no way to auto-detect that (Python packaging environment markers only cover OS/architecture, not GPU vendor), so ROCm support is provided as an **opt-in extra** instead, sourced entirely as prebuilt wheels from AMD's own package index (no source builds).

This has been verified end-to-end on an AMD Radeon 880M/890M (Ryzen AI APU, `gfx1150`) on Arch-based Linux (Omarchy), producing a working `torch.cuda.is_available() == True` (PyTorch's ROCm builds reuse the CUDA API namespace) and a successful GPU matrix multiply.

&nbsp;

## Overview: the "pre-setup" scripts

Two scripts support this workflow:

| Script | Purpose |
| --- | --- |
| `scripts/detect_rocm_gpu.py` | **Preflight check.** Detects your AMD GPU's `gfx` architecture target and validates driver/permissions/library prerequisites, *before* you install anything. Read-only, no ROCm install required. |
| `scripts/reset_rocm_setup.sh` | **Teardown / reset.** Reverses the venv-scoped ROCm install and reports any OS-level changes (group membership, udev rules) that may need manual cleanup. |

&nbsp;

## 1. Run the preflight check

From the repository root:

```bash
python scripts/detect_rocm_gpu.py
```

This script reads the Linux kernel's `amdgpu`/KFD driver directly (no ROCm userspace install required) and reports:

- Whether the `amdgpu` kernel driver is loaded
- Each detected AMD GPU's **gfx architecture target** (e.g. `gfx1150`, `gfx1100`, `gfx942`), decoded from `/sys/class/kfd/kfd/topology/nodes/*/properties`
- Whether `/dev/kfd` and `/dev/dri/renderD*` are accessible (either directly world-accessible, or via `render`/`video` group membership -- see [Prerequisites](#prerequisites-for-other-machines) below if not)
- Whether required libraries (`libatomic`, `libquadmath`) are present
- Whether your Python version is in ROCm's officially supported range (3.11-3.14; see the [Python version caveat](#python-version-caveat) below for this project specifically)

Example output on a Ryzen AI APU with an integrated Radeon 880M/890M GPU:

```
OK: GPU node 1: gfx_target_version=110500 -> gfx1150 (simd_count=32)
Suggested pyproject.toml extra: device-gfx1150
```

Exit code `0` means all checks passed; `1` means a hard blocker (no driver, no GPU node, unsupported OS); `2` means the GPU was found but there are warnings to review (permissions, missing libs).

&nbsp;

## 2. Match the extra in `pyproject.toml`

The `[project.optional-dependencies].rocm` group in `pyproject.toml` pins `torch`, `torchvision`, and `torchaudio` (plus their transitive ROCm SDK dependencies -- see [How this works](#how-this-works) below) to a specific `device-gfxNNNN` extra and ROCm/PyTorch version. If the preflight script suggests a different `gfx` target than what's currently pinned, update every `device-gfx1150` occurrence in the `rocm = [...]` list and `[tool.uv.sources]` to match your GPU (or use the broader `device-all` extra, which is larger but works across GPU families -- see AMD's index for available `rocm-sdk-device-*` package names).

&nbsp;

## 3. Install into a separate ROCm environment

<a id="python-version-caveat"></a>

**Python version caveat:** the ROCm extra requires **Python < 3.14**. This isn't an AMD limitation (their index does publish `cp314` wheels) -- it works around a `uv` universal-lock resolver limitation where marker-forks spanning Python 3.14 and non-Linux platforms merge incorrectly and produce spurious "no solution found" errors. If your default `uv python` is 3.14 (check with `uv run python --version`), pass `--python 3.12` to the `uv sync` commands below.

This repository keeps **two environments side by side**, both generated from this single `pyproject.toml` and `uv.lock` (only the target directory differs -- there is no second project file):

| Environment | Contents | Use it for |
| --- | --- | --- |
| `.venv` (default) | PyPI PyTorch (CPU/NVIDIA CUDA) **+ TensorFlow** | everything the book does by default, plus loading OpenAI's original GPT-2 checkpoints |
| `.venv-rocm` | ROCm PyTorch from AMD's index, **no TensorFlow** | running the code on your AMD GPU |

```bash
uv sync --dev                       # .venv  (default, unchanged behavior)

UV_PROJECT_ENVIRONMENT=.venv-rocm \
  uv sync --dev --extra rocm --no-install-package tensorflow-cpu
```

Or, with the [mise](https://mise.jdx.dev) tasks defined in `mise.toml`:

```bash
mise run sync          # .venv
mise run sync:rocm     # .venv-rocm
```

The ROCm sync downloads several GB of wheels (`torch`, `triton`, `rocm-sdk-core`, `rocm-sdk-device-gfx1150`, etc.) -- expect this to take a while on the first run. Everything is installed into `.venv-rocm` only; **no OS packages are touched**.

TensorFlow is deliberately excluded from the ROCm environment -- see [TensorFlow and ROCm PyTorch](#tensorflow-and-rocm-pytorch) below for why.

&nbsp;

### Running code in the ROCm environment

Because the `rocm` extra is declared as conflicting with the default `cuda` extra (`[tool.uv].conflicts` in `pyproject.toml`), a bare `uv run` always syncs the *default* environment. To run against the AMD GPU, either select the ROCm environment explicitly:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync python your_script.py
# or: mise run rocm python your_script.py
```

or activate it once:

```bash
source .venv-rocm/bin/activate
python your_script.py
```

> **Note:**
> Pass `UV_PROJECT_ENVIRONMENT` the plain directory name (`.venv-rocm`). uv resolves it relative to the *project root*, so it works unchanged from any chapter subdirectory. Passing a path relative to your current directory (e.g. `../../.venv-rocm` from `ch05/01_main-chapter-code`) makes uv create a stray environment outside the repository.

&nbsp;

### Jupyter / VS Code kernels

Register one kernel per environment so notebooks can be switched between CPU/CUDA and AMD GPU from the kernel picker:

```bash
mise run kernels
```

which is equivalent to:

```bash
uv run --no-sync python -m ipykernel install --user \
  --name llms-cpu --display-name "LLMs (CPU/CUDA + TF)"
UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync python -m ipykernel install --user \
  --name llms-rocm --display-name "LLMs (AMD ROCm)"
```

&nbsp;

> **Note:**
> The main chapter notebooks in this repository use `torch.cuda` (and `torch.mps`) device checks, which work unchanged on ROCm builds of PyTorch -- ROCm reuses the `torch.cuda` API namespace, so `device = "cuda"` selects your AMD GPU.

&nbsp;

## 4. Verify

```bash
UV_PROJECT_ENVIRONMENT=.venv-rocm uv run --no-sync python -c "
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device='cuda')
print((x @ x).sum().item())
"
```

Expected output (device name will match your GPU):

```
2.13.0+rocm10.0.0 True
AMD Radeon 890M Graphics
<some number>
```

`mise run rocm:check` runs the preflight script and this GPU check together.

&nbsp;

<a id="tensorflow-and-rocm-pytorch"></a>

## TensorFlow and ROCm PyTorch

This repository lists `tensorflow`/`tensorflow-cpu` as a dependency, used *only* by the `gpt_download.py` scripts that load OpenAI's original GPT-2 weights (ch05/ch06/ch07 and appendix-E). TensorFlow and ROCm's PyTorch stack each bundle their own LLVM, and loading both into one process aborts it:

```
: CommandLine Error: Option 'print-inst-addrs' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

Importing TensorFlow *before* torch happens to work, but the repository's own code hits the bad order in practice: the ch05/ch06/ch07 notebooks `import torch` early and only later do `from gpt_download import download_and_load_gpt2`; `pkg/llms_from_scratch/ch05.py` imports torch at module level and TensorFlow lazily inside `download_and_load_gpt2()`; and `setup/02_installing-python-libraries/python_environment_check.py` imports packages in `requirements.txt` order. (`ch05/01_main-chapter-code/gpt_generate.py` imports TensorFlow first and is unaffected.)

Rather than relying on import-order discipline, the `.venv-rocm` environment simply has no TensorFlow in it, which makes the crash impossible. To still use the original GPT-2 weights on the GPU, convert them once in the default environment:

```bash
uv run --no-sync python scripts/export_gpt2_params.py --model-size 124M
# or: mise run gpt2:export
```

This downloads the checkpoint (if needed) and writes `gpt2/124M_params.npz` (498 MB for the 124M model). In the ROCm environment, load it with NumPy only. In the ch05 notebook, this replaces the single cell that reads

```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

with

```python
import sys
sys.path.insert(0, "../../scripts")          # from ch05/01_main-chapter-code
from export_gpt2_params import load_params

settings, params = load_params("../../gpt2/124M_params.npz")
```

Everything after that -- `load_weights_into_gpt(gpt, params)`, `gpt.to(device)`, `generate(...)` -- is unchanged book code. Verified end to end on this machine: GPT-2 124M weights loaded in `.venv-rocm` and generating text on `AMD Radeon 890M Graphics`.

Note that `ch05/01_main-chapter-code/gpt_generate.py` imports TensorFlow at module level, so that particular script only runs in the default environment; the notebook and the `llms_from_scratch` package both work TensorFlow-free with the `.npz` route above.

The bonus material in [../../ch05/02_alternative_weight_loading](../../ch05/02_alternative_weight_loading) is another TensorFlow-free route, loading equivalent weights via `safetensors`/`transformers`.

&nbsp;

## How this works

AMD publishes ROCm-enabled PyTorch wheels across **two separate simple package indexes**:

- `https://stable.repo.amd.com/rocm/pytorch/whl-next/` -- `torch`, `torchvision`, `torchaudio`, `triton`, and the per-GPU `amd-torch-device-gfxNNNN` packages.
- `https://stable.repo.amd.com/rocm/core/whl-next/` -- the ROCm SDK itself: `rocm`, `rocm-bootstrap`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfxNNNN`.

Both are configured as `explicit = true` indexes in `[[tool.uv.index]]`, routed via `[tool.uv.sources]`, gated to Linux/non-aarch64/Python<3.14 by environment markers. Because they are `explicit`, they are only ever consulted for the packages routed to them -- a default `uv sync` (without `--extra rocm`) still resolves the regular PyPI `torch` build, so CPU/CUDA/MPS users and CI are unaffected.

One non-obvious detail drove the final working configuration: **`tool.uv.sources` only applies to *direct* project dependencies.** `torch` transitively depends on `rocm[libraries]`, which transitively depends on `rocm-sdk-libraries`, etc. -- but uv silently ignores source routing for anything not listed directly in `pyproject.toml`. That's why every package in that dependency chain (`rocm`, `rocm-bootstrap`, `rocm-sdk-core`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1150`, `triton`, `amd-torch-device-gfx1150`, `amd-torch-device-gfx115x`, `amd-torchvision-device-gfx1150`) is listed explicitly in the `rocm` extra, pinned to the exact version AMD publishes (important, since an unrelated package named `rocm-bootstrap` also exists on PyPI).

&nbsp;

> **Note:**
> If your project sets a repo-wide `exclude-newer` freshness cutoff under `[tool.uv]`, you also need `exclude-newer-package = { torch = false, ... }` for every package above: AMD's index wheels don't publish upload-date metadata, so uv can't prove they satisfy the cutoff and excludes them. The boolean `false` fully disables the cutoff for that package name (substituting a different *date* does not help). This repository does not set `exclude-newer`, so no such override is needed here.

If you'd rather use PyTorch.org's own official `rocm7.x` channel instead of AMD's index: as of this writing, that channel publishes only one `torch` build per ROCm version, whose `rocm` stub dependency is published there as a **source-only tarball** (no wheel), which breaks a wheels-only install. AMD's own index was used instead for that reason.

&nbsp;

## Tearing down / resetting

To undo the ROCm setup:

```bash
scripts/reset_rocm_setup.sh          # re-syncs the venv without the rocm extra
scripts/reset_rocm_setup.sh --full   # also prunes uv's cache of the AMD wheels
scripts/reset_rocm_setup.sh --dry-run  # preview only, changes nothing
```

This script re-runs `uv sync` (without `--extra rocm`) to restore the default CPU/CUDA torch build in `.venv`, and reports (without automatically changing) any OS-level state the general ROCm setup process can involve on other machines: `render`/`video` group membership, a `/etc/udev/rules.d/70-amdgpu.rules` file, and `libatomic`/`libquadmath` system packages. On this project's own verified dev machine, none of these OS-level changes were actually necessary (see below), so there was nothing to remove there -- but the script still checks and reports, since your machine may differ.

&nbsp;

## Prerequisites for other machines

<a id="prerequisites-for-other-machines"></a>

AMD's official prerequisite docs are validated against Ubuntu/RHEL/SLES; other distros (e.g. Arch-based) aren't officially tested, though the pip wheels bundle their own ROCm runtime libraries and are generally distro-agnostic as long as the following are true:

- The `amdgpu` kernel driver is loaded (`lsmod | grep amdgpu`).
- Your user can access `/dev/kfd` and `/dev/dri/renderD*` -- either because they're world-accessible, or because your user is in the `render` and/or `video` group (`sudo usermod -aG render,video "$USER"`, then log out/in), or via a udev rule granting access.
- `libatomic` and `libquadmath` are installed (commonly part of your distro's base GCC runtime libraries already).
- A supported Python version (3.11-3.14 per AMD; this project's `rocm` extra additionally requires **< 3.14**, see above).

`scripts/detect_rocm_gpu.py` checks all of these for you.

&nbsp;

## Questions?

If you have any questions, please don't hesitate to reach out via the [Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) forum in this GitHub repository.
