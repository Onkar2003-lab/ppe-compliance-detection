# PPE-Compliance Detection — Benchmark & Deployment-Readiness Study

Reproducible code for an MSc dissertation: benchmarking three YOLO generations
(**YOLOv8 · YOLO11 · YOLO26**) for PPE-compliance detection on a four-axis
**deployment-readiness** framework, with a cross-dataset (SH17 ↔ CHV) transfer study and a
real-time zone-compliance demonstration.

Every number in the dissertation is produced by a command in this repository. §5 maps each
reported result to the command that regenerates it; §4 maps each run-ID to its frozen config
and its outputs. Nothing is reported that cannot be traced back through those two tables.

> Vault brain for this repo: `04-Experiments/00-coding-context.md` (scoped requirements) and
> `04-Experiments/00-Coding-Hub.md` (front door). Method: `99-Playbooks/x-experiment-loop.md`.

---

## Repo layout

```
code/
  README.md              ← this file: env · data · how to run · run-ID map · finding map
  requirements.txt       ← direct deps (torch installed separately — see §1)
  requirements.lock      ← the full pinned set the results were produced with
  configs/
    base.yaml            ← training template; every run config is derived from it
    data/                ← Ultralytics dataset YAMLs (sh17, chv, and their -640 variants)
    splits/              ← THE FROZEN SPLITS: six image lists + split-manifest.json
    trackers/            ← ByteTrack settings for the demo
    X03-*.yaml X04-*.yaml← one frozen config per run-ID (18 training runs + the timing pilot)
  scripts/
    verify_env.py        ← GPU / CUDA / dataset sanity check (run this first)
    make_demo_clip.py    ← renders stills into a clip, for latency measurement only
  src/                   ← the pipeline (§3)
  tests/                 ← 193 tests; the logic that decides a claim is tested, not sampled
  notebooks/             ← jupytext-paired exploration (outputs stripped)
```

**Heavy artefacts live OUTSIDE this repo and are gitignored:** datasets → `D:\Dissertation\`,
run outputs and weights → `D:\runs\`. Only code, configs and split lists are versioned.

---

## 1. Environment setup (Windows / PANDORA — RTX 4070 Laptop, 8 GB)

The RTX 4070 Laptop is Ada-generation (CUDA 12.x). **The default `pip install torch` often
pulls a CPU-only build on Windows** — install the CUDA build explicitly first.

```powershell
# from the code/ folder, in PowerShell
py -3.13 -m venv .venv           # built on Python 3.13 (3.10–3.13 all fine)
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) CUDA-enabled PyTorch FIRST (cu124 build — matches Ada / CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) then the rest
pip install -r requirements.txt          # or requirements.lock to pin exactly (see below)

# 3) register the Jupyter kernel for this env
python -m ipykernel install --user --name ppe --display-name "Python (ppe)"

# 4) verify the GPU is actually visible before any training
python scripts/verify_env.py     # exits 0 only if CUDA + both dataset roots are present
```

`verify_env.py` must log `CUDA available: True` and `GPU: NVIDIA GeForce RTX 4070 Laptop GPU`,
and find both dataset roots (from `configs/base.yaml`). It exits non-zero otherwise. If CUDA is
`False`, a CPU-only torch got installed — uninstall torch/torchvision and redo step 1.

> If `Activate.ps1` trips the execution policy in a fresh window, call the interpreter directly:
> `.venv\Scripts\python.exe -m src.score`. Every command below works either way.

> **Clone to a short path on Windows** (`D:\ppe-compliance-detection`, not somewhere nested deep in
> `AppData\Local\Temp\…`). The pinned JupyterLab widget extensions carry filenames that exceed
> `MAX_PATH`, and pip aborts mid-install with `OSError: [Errno 2] No such file or directory` on a
> `…webpack_sharing_consume_default_jquery…js`. Nothing to do with torch or the pipeline — the
> notebook stack is in the lock because the environment was frozen wholesale.

### Frozen versions (env built 2026-07-25 — repro-critical)

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

`requirements.lock` is the full pinned set (committed on purpose — it is the reproducibility
contract). `pip install -r requirements.lock` after step 1 reproduces the exact environment.

---

## 2. Data

### 2.1 The three datasets

Roots are set in `configs/base.yaml` (`sh17_root` / `chv_root`); `verify_env.py` asserts them.
**No dataset file is committed here** — only the split lists that name the images.

**SH17** (primary training set) — CC BY-NC-SA 4.0, non-commercial, academic use fine:
- `kagglehub.dataset_download("mugheesahmad/sh17-dataset-for-ppe-detection")`;
  mirror/metadata https://github.com/ahmadmughees/SH17dataset.
- Source classes used: `person=0`, `helmet=10`, `safety-vest=16`.

**CHV** (cross-test set) — https://github.com/ZijianWang-ZW/PPE_detection.
- 6 classes: helmet ×4 colours, person, vest.

**Pictor-PPE** (violation axis, **evaluation only**) — never trained on; `src/guards.py`
enforces this in code, so a config that tries to train on it fails rather than quietly
contaminating the transfer study.

### 2.2 The harmonised space and the frozen splits

`src/harmonise.py` maps both datasets into the shared label space **`{0: person, 1: helmet,
2: vest}`** and freezes the splits. It writes to `D:/Dissertation/harmonised/`, excluding the
SH17↔CHV near-duplicates found at X01/S1.4 (`configs/splits/X01-cross-dataset-duplicates.yaml`),
so a cross-dataset score cannot be inflated by an image the model met in training.

**The exact splits every reported number was scored on are committed** in `configs/splits/`:

| Dataset | Split | Images | Split ID |
|---|---|---|---|
| SH17 | train | 5,832 | `5707d57d1e65` |
| SH17 | val | 647 | `3a8265901147` |
| SH17 | test | 1,615 | `232dcc019591` |
| CHV | train | 1,064 | `18743a2bd145` |
| CHV | val | 130 | `9e317df0520c` |
| CHV | test | 132 | `f3060bfbab12` |

The split ID is `sha1(sorted image stems)[:12]`, recorded in `configs/splits/split-manifest.json`
alongside the class mapping, the validation seed and the exclusion counts. The lists hold the
absolute paths training actually read; the ID depends only on the filenames, so it survives being
checked out anywhere. `tests/test_harmonise.py` recomputes all six IDs from the committed lists
on every test run — a list cannot drift from the manifest without failing the suite.

**val is not test.** Training evaluates on **val** (it drives early stopping and `best.pt`
selection). Every headline number is scored on **test**, which no training run has ever read.

### 2.3 Rebuilding the data (only if starting from raw downloads)

```bash
python -m src.harmonise      # raw datasets -> D:/Dissertation/harmonised/ + split-manifest.json
python -m src.preresize      # -> D:/Dissertation/harmonised-640/ (the I/O fix, §3)
```

`src/preresize.py` resizes the images offline to the training resolution. This is a **data-path**
optimisation only: the pixels the model trains on are unchanged, and it made the pipeline 6.1×
faster after training was found to be I/O-bound with the GPU idle at 0 %.

---

## 3. The pipeline

Each module is one stage and one entry point (`python -m src.<module>`). Later stages read
earlier stages' artefacts and never recompute them, so a figure cannot drift from a ledger row.

| Stage | Command | Produces |
|---|---|---|
| Label audit | `python -m src.audit_labels` | SH17/CHV label-file audit (X01) |
| Pictor audit | `python -m src.audit_pictor` | the eval-only violation set, audited |
| EDA / domain shift | `python -m src.eda` | class balance, resolution, co-occurrence, shift |
| Near-duplicates | `python -m src.overlap` | the cross-dataset exclusion list |
| Harmonise | `python -m src.harmonise` | shared label space + frozen splits |
| Pre-resize | `python -m src.preresize` | 640-px training copy |
| Association rule | `python -m src.associate` | person↔PPE containment rule (τ = 0.80) |
| **Train one run** | `python -m src.run --config configs/<run-id>.yaml` | weights + val summary |
| **Train the grid** | `python -m src.grid run` | all 18 runs, resumable queue |
| Grid state | `python -m src.grid status` | what is done, pending, failed |
| **Accuracy + transfer** | `python -m src.score` | test-split scores, all 18 runs |
| **Violation recall** | `python -m src.violation` | zero-shot compliance on Pictor |
| **Efficiency** | `python -m src.efficiency` | latency / throughput, GPU and CPU |
| **Aggregate + test** | `python -m src.stats` | mean ± 95 % BCa CI, sign-flip, Friedman/Nemenyi |
| **Figures** | `python -m src.figures` | the six-figure set, PNG + vector PDF |
| Demo (headless) | `python -m src.monitor --config configs/demo.yaml` | violation log + snapshots |
| Demo evaluation | `python -m src.demo_eval` | alert accuracy by zone |
| Draw a zone | `python -m src.zone --source <video>` | `zone.yaml` (interactive window) |
| **Demo console** | `python -m src.dashboard` | http://127.0.0.1:8000 |

**S5 order matters** — run `score` / `violation` / `efficiency` before `stats`, and `stats`
before `figures`.

### The demonstration console

`python -m src.dashboard` serves a local page: choose a source (video file, webcam index, or
RTSP/MJPEG URL), drag a box to mark the zone where PPE is mandatory, watch the annotated stream,
read the violation log, download the CSV. It drives `src.monitor.iterate` frame by frame rather
than re-implementing anything, so the console and the reported numbers cannot disagree. Nothing
leaves the machine.

---

## 4. Run-ID ↔ config ↔ outputs

**The contract:** run-ID `X##-<model>-s<seed>-<dataset>`. One vault ledger row per run **before**
launch; the config frozen at `configs/<run-id>.yaml` with the commit SHA in the row; the seed
lives in the config. **No config change without a new run-ID** — a frozen config is never edited.

The run-ID determines everything else:

| Run-ID | Frozen config | Outputs | Test scores |
|---|---|---|---|
| `X04-<model>-s<seed>-<dataset>` | `configs/X04-<model>-s<seed>-<dataset>.yaml` | `D:/runs/X04-<model>-s<seed>-<dataset>/` (`weights/best.pt`, `summary.json`) | `D:/runs/X05-accuracy/<run-id>/` |

with **model** ∈ `y8n` (YOLOv8n) · `y11n` (YOLO11n) · `y26n` (YOLO26n), **seed** ∈ `0, 1, 2`,
**dataset** ∈ `sh17, chv` — 3 × 3 × 2 = **18 runs**, the complete grid.
`python -m src.grid status` enumerates all 18 with their state.

`X03-yolov8n-s0` and `X03-retime-yolov8n-s0` are the timing pilot and its re-time after the I/O
fix. They are cost measurements, not results.

**Fixed across every run:** imgsz 640 · batch 16 (nano) · `amp=True` · 200 epochs with early-stop
patience 50 · `cache='disk'` · workers 4–8. Only the model, seed and dataset vary.

**Two caveats a reader should know**, both recorded in the vault ledger: a resumed run's
`train_seconds` covers only the resumed fragment, so it is never quoted as a cost (scores are
unaffected — `src/run.py` evaluates `best.pt`); and resumption restarts the patience counter, which
is why one run overran its stopping point.

---

## 5. Finding → command

Where each reported result comes from. Numbers are canonical in the vault
(`04-Experiments/run-ledger.md` and `00-Key-Facts.md`); this table says how to regenerate them.

| Reported result | Regenerate with | Artefact |
|---|---|---|
| In-domain accuracy, cross-dataset transfer (test splits) | `python -m src.score` | `D:/runs/X05-accuracy/X05-accuracy-per-run.json` |
| Violation recall; person detection as the bottleneck | `python -m src.violation` | `D:/runs/X05-violation/` |
| Latency / throughput, GPU and CPU; GFLOPs vs real speed | `python -m src.efficiency` | `D:/runs/X05-efficiency/` |
| Seed aggregation (mean ± 95 % BCa CI), sign-flip and Friedman/Nemenyi tests | `python -m src.stats` | `D:/runs/X05-stats/` |
| The six figures | `python -m src.figures` | `D:/runs/X05-figures/` (PNG + PDF) |
| Demo alert accuracy by zone | `python -m src.demo_eval` | `D:/runs/X06-demo-eval/` |
| Demo end-to-end latency (GPU / CPU) | `python -m src.monitor --config configs/demo.yaml --out <dir> --no-display` (add `--device cpu` for the CPU case) | `demo-metrics.json` |
| Zone membership, dwell and debounce behaviour | `pytest tests/test_zone.py tests/test_dwell.py` | 23 cases, exact |

### Verifying a number from a clean checkout — done, and it passes

Evaluation is deterministic: re-scoring a set of weights on a frozen test split returns the same
number every time. So the release check is one run of the accuracy axis:

```bash
git clone <this repo> && cd ppe-compliance-detection
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe -m src.score --pattern X04-y8n-s2-chv --out <scratch-dir>
```

Run on **2026-08-11** from a fresh clone and a fresh environment built from `requirements.lock`,
against `X04-y8n-s2-chv` — chosen because it trained start to finish with no resume, so nothing
about it is qualified. **All 16 reported values came back bit-identical** to
`D:/runs/X05-accuracy/X05-accuracy-per-run.json`: both mAP50 and mAP50-95, precision and recall,
all six per-class scores, and both transfer deltas. In-domain mAP50 `0.9009138488046622`,
cross-domain `0.5340983294274633`, transfer delta `-0.3668`, to the last digit.

The weights themselves are not distributed (§7); the check needs `D:/runs/X04-*/weights/best.pt`
and the harmonised data on disk. What it establishes is that the released code, configs and split
lists turn those weights into exactly the reported numbers — no hidden state on the machine that
produced them.

⛔ **Training reproduction is a stronger claim and is not made.** cuDNN and AMP are
non-deterministic on this hardware, so a retrained run would land near its original, not on it.

---

## 6. Reproducibility contract

- Run-ID scheme + frozen config per run + commit SHA in the ledger row (§4).
- Frozen splits committed and self-checked by the test suite (§2.2).
- Every library version pinned in `requirements.lock` (§1).
- Numbers graduate to `00-Key-Facts.md` **only** seed-aggregated: mean ± 95 % BCa CI with a named
  test. **No "X beats Y" without the test** — exact paired sign-flip for pairwise comparisons,
  Friedman with Nemenyi post-hoc across models.
- Pictor-PPE is evaluation-only, enforced in `src/guards.py`.
- `pytest` before any push: **193 tests** in this released checkout, `ruff` and `black` clean (line
  width 100, pinned in `pyproject.toml`). The development tree runs 207: the extra 14 cover
  `src/keepawake.py`, which holds this particular laptop out of Modern Standby during a long
  training queue. It guards one machine's power behaviour rather than the method, so it is
  deliberately not released; `src/grid.py` imports it inside a `try`/`except` and runs without it.

## 7. Licence and citation

Code released under the **MIT Licence** (see `LICENSE`). The datasets are **not** redistributed
here and keep their own terms — SH17 is CC BY-NC-SA 4.0; CHV and Pictor-PPE are governed by their
respective sources (§2.1). Trained weights are not distributed either; they are regenerable from
the frozen configs.

If this code is useful, please cite the dissertation (MSc, WMG, University of Warwick, 2026) and
the originating dataset papers.
