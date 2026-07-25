# PPE-Compliance Detection — Benchmark & Deployment-Readiness Study

Reproducible code for an MSc dissertation: benchmarking three YOLO generations
(**YOLOv8 · YOLO11 · YOLO26**) for PPE-compliance detection on a 4-axis
deployment-readiness framework, with a cross-dataset (SH17 ↔ CHV) transfer study and a
real-time zone-compliance demo.

> Vault brain for this repo: `04-Experiments/00-coding-context.md` (scoped requirements)
> and `04-Experiments/00-Coding-Hub.md` (front door). Method: `99-Playbooks/x-experiment-loop.md`.

---

## Repo layout
```
code/
  README.md            ← this file (env setup + data + how to run)
  requirements.txt     ← Python deps (torch installed separately — see below)
  .gitignore .gitattributes
  configs/
    base.yaml          ← training config template (copied per run-id to ../configs/)
  scripts/
    verify_env.py      ← GPU / CUDA / ultralytics sanity check (run this first)
  src/                 ← (built per experiment card: harmonise / train / evaluate / stats)
  demo/                ← (built later: ByteTrack + per-person PPE state + zone logging)
```
**Heavy artefacts live OUTSIDE this repo and are gitignored:**
datasets → `D:\Dissertation\` (SH17_dataset / CHV_dataset), run outputs/weights → `D:\runs\`. Only code + configs are versioned.

---

## 1. Environment setup (Windows / PANDORA — RTX 4070 Laptop, 8 GB)

The RTX 4070 Laptop is Ada-generation (CUDA 12.x). **The default `pip install torch`
often pulls a CPU-only build on Windows** — install the CUDA build explicitly first.

```powershell
# from the code/ folder, in PowerShell
py -3.13 -m venv .venv           # built on Python 3.13 (only version on PANDORA; 3.10–3.13 fine)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) CUDA-enabled PyTorch FIRST (cu124 build — matches Ada / CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) then the rest
pip install -r requirements.txt

# 3) register the Jupyter kernel for this env
python -m ipykernel install --user --name ppe --display-name "Python (ppe)"

# 4) verify the GPU is actually visible before any training
python scripts/verify_env.py     # exits 0 only if CUDA + both dataset roots are present
```
`verify_env.py` must log `CUDA available: True` and `GPU: NVIDIA GeForce RTX 4070 Laptop GPU`,
and find both dataset roots (from `configs/base.yaml`). It exits non-zero otherwise.
If CUDA is `False`, a CPU-only torch got installed — uninstall torch/torchvision and redo step 1.

### Frozen versions (env built 2026-07-25 — repro-critical, mirror to `00-Key-Facts` + `00-coding-context §2`)
| Component | Version |
|---|---|
| Python | 3.13.14 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torch CUDA build | 12.4 |
| CUDA driver | 610.47 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB) |
| ultralytics | 8.4.105 |
| numpy / scipy | 2.4.4 / 1.18.0 |
| scikit-posthocs | 0.14.0 |
| opencv-python | 5.0.0.93 |
| jupytext / nbstripout | 1.19.5 / 0.9.1 |

Full pinned set: `requirements.lock` (committed). Regenerate with `pip freeze > requirements.lock`.

---

## 2. Datasets (✅ ON DISK 2026-07-25 — sources verified)

Roots are set in `configs/base.yaml` (`sh17_root` / `chv_root`); `verify_env.py` asserts them.
Do **not** re-download. ⚑ Confirm the internal structure (images / labels / `sh17.yaml`) at the X01 audit.

**SH17** (primary training set) — CC BY-NC-SA 4.0 (non-commercial; fine for academic use):
- Location: **`D:\Dissertation\SH17_dataset`**.
- Source (for reference): `kagglehub.dataset_download("mugheesahmad/sh17-dataset-for-ppe-detection")`;
  mirror/metadata https://github.com/ahmadmughees/SH17dataset (`sh17.yaml`).
- Core classes we use: `person=1`, `safety-vest=13`, `helmet=15` (+ worn-state on/off tags).

**CHV** (cross-test set):
- Location: **`D:\Dissertation\CHV_dataset`**.
- Source (for reference): https://github.com/ZijianWang-ZW/PPE_detection (images via its Google Drive link).
- 6 classes: helmet ×4 colours, person, vest.

> ⚠️ Do **not** commit dataset files. The dataset folders live outside the repo and are gitignored.
> Record only split IDs + manifest hashes in the vault run-ledger.

The first experiment (**X01**) audits the actual label files of both sets before any
training — confirms the taxonomy overlap (closes assumption A1), decides the violation
route, and checks SH17/CHV parent-set overlap (both trace to SHWD/GDUT-HWD).

---

## 3. Reproducibility contract
- Run-ID scheme `X##-<model>-s<seed>`; one vault ledger row per (model × seed) **before** launch.
- Config frozen to `../configs/<run-id>.yaml` + commit SHA in the row; seeds live in the config.
- Fixed: imgsz 640, batch n=16/s=8, `amp=True`, 200 epochs + early-stop patience,
  `cache='disk'`, workers 4–8. No config change without a new run-ID.
- Numbers graduate to `00-Key-Facts.md` only seed-aggregated (mean ± 95 % CI + named test).

## 4. Git workflow
Own git repo here → GitHub (user account). Claude edits + stages with conventional commit
messages at each milestone; **user runs `git push`** (credentials stay user-side).
