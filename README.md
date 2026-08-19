# ilmu-glossary

**Routing-aware quantization calibration for Bahasa Melayu.**

Does post-training quantization calibrated on non-Malay text impose a
measurable, recoverable accuracy penalty on Bahasa Melayu workloads — and does
routing-coverage-driven calibration sample selection recover it better than
corpus substitution alone?

Target model: **NVIDIA Nemotron 3.5 Lightning 30B-A3B** (hybrid Mamba-2 + MoE
+ attention). Compute: **1× B200 on Modal**.

> This studies **open base checkpoints, not ILMU's fine-tune.**
> Architecture-determined findings transfer; model-quality findings do not.
> No claim is made about ILMU accuracy.

The original spec's stated mechanism rests on precision assignments that the
shipped checkpoint does not use, which is why two recipe families are run
rather than one. The deviations from that spec are recorded outside this
repository; source comments refer to them by id (D1, D2, ...).

---

## The question, precisely

Nemotron 3.5 Lightning's NVFP4 checkpoint quantizes routed and shared experts
to 4 bits. Its supported languages are English, Spanish, French, German,
Italian, Japanese and code. **Malay is not among them**, and NVIDIA's
calibration set does not contain it.

If Malay routes through a different subset of experts than the calibration
corpus's languages do, those experts receive sparse calibration coverage. The
question is whether that costs measurable Malay accuracy, and whether
choosing calibration samples to maximise expert coverage recovers it.

Prior art cuts both ways — on a SEA multilingual MoE, vanilla FP8 degraded
Thai −10.17% and Vietnamese −11.75% while an expert-aware scheme recovered to
within −2.5%; other work found k-quantization does *not* disproportionately
harm multilingual performance. Hence measurement rather than assumption.

## Two recipe families

| Family | Expert precision | What it tests |
| --- | --- | --- |
| `w4a16_shipped` | NVFP4 W4A16, weights only | **Ecological validity.** NVIDIA's published recipe, unchanged except for calibration data. Calibration influences the checkpoint only through the static MSE precision search. |
| `w4a4_mechanism` | NVFP4 W4A4, per-tensor activation scales on routed experts | **Mechanism validity.** The configuration in which sparse routing can actually starve amax estimation. **NVIDIA does not ship this.** |

Running both separates *"calibration language matters for FP4 MoE in
principle"* from *"calibration language matters for the checkpoint you can
download today."*

## Pipeline

| Phase | Module | Runs on | Produces |
| --- | --- | --- | --- |
| 0 | `data_prep.py` | CPU | Six corpus classes + parallel BM/EN corpus, 80/20 splits, `corpus_stats.parquet` |
| 1 | `routing_analysis.py` | B200 | Per-class expert activation, entropy, Jaccard, KL, coverage curves |
| 2 | `calib_select.py` | CPU | Six calibration variants × four sample counts, coverage statistics |
| 3 | `quantize.py` | B200 | Quantized checkpoints, `quantization_runs.parquet` |
| 4 | `evaluate.py` | B200 | Four evaluation tiers + AIPerf throughput |
| 5 | `analyze.py` | CPU | `REPORT.md` |

Every artifact persists to a Modal Volume. No phase re-runs a previous phase.

### Evaluation tiers are not interchangeable

| Tier | Instrument | Controls for | Carries |
| --- | --- | --- | --- |
| 1 | KL divergence on **parallel** BM/EN pairs | Content, domain, difficulty | **The causal claim** |
| 2 | Perplexity on held-out slices | Distribution | Magnitude |
| 3 | Cross-MMLU (parallel) | Content, difficulty | Downstream accuracy |
| 4 | MalayMMLU (native) | Nothing | **The relevance argument** |

Tier 1 is primary because content and difficulty are held constant across each
aligned pair, so a non-zero **BM−EN delta** is attributable to language rather
than register. Tier 4 is a coarse detector and is treated as a secondary
signal — published work found a 1.7% automatic-metric drop corresponding to
16.0% under human evaluation.

## Setup

```bash
uv sync --extra dev            # host deps; torch/vllm live in the Modal image
modal token new                # once
modal secret create huggingface-secret HF_TOKEN=hf_...
```

Nothing CUDA-bound installs on the laptop. `modal_app/images.py` builds the
GPU image with torch 2.9 / vLLM 0.27.1 / ModelOpt 0.40 on CUDA 12.8.

## Running

Validate ids and config without spending anything:

```bash
uv run ilmu preflight
```

End-to-end smoke test on a small MoE, tiny N — proves every phase's plumbing
for a few dollars before any real B200 spend:

```bash
modal run modal_app/app.py --dry-run
```

The real staged run:

```bash
modal run modal_app/app.py --config configs/config.yaml
```

Individual phases:

```bash
modal run modal_app/app.py::phase0_data_prep
modal run modal_app/app.py::phase1_routing
```

### The staged run stops on its own

`oracle_contaminated` is built and evaluated **first**, at N=1024, before the
rest of the matrix. It is deliberately calibrated on the evaluation
distribution and **is not a legitimate result** — it exists to establish the
upper bound on what any calibration strategy could recover.

- If `oracle_contaminated` ≈ `baseline_en`, calibration is not the lever. The
  pipeline records that, writes it as the finding, and **stops** rather than
  burning 24 PTQ runs.
- If the gap is large, the full matrix proceeds. `coverage_greedy` approaching
  oracle is the success condition.

## Guarantees the code enforces

These are assertions, not conventions:

- **Contamination.** 80/20 splits are persisted with a fixed seed and
  disjointness is asserted before phases 2, 3 and 4. `oracle_contaminated`
  must pass `allow_contamination=True` explicitly — the check cannot be
  skipped, only acknowledged.
- **Recipe identity.** Variants within a family must hash to the same recipe.
  If anything but the calibration data source differs, `quantize.py` refuses
  to run.
- **Provenance.** Every parquet carries the config fingerprint, git sha and
  write time. Two artifacts with different fingerprints were not produced by
  the same settings.
- **Contamination labelling.** `CalibVariant.is_contaminated` drives the label
  in every table and caption, so it cannot be forgotten in one of them.
- **Malay ≠ Indonesian.** Three-layer filter (Malaysian-source allowlist →
  mesolitica fastText ms/id discrimination → discriminative lexicon), with a
  100-document-per-class spot-check logged.

## Reporting rules

- A null result is reported plainly, in the abstract, not a footnote.
- Every number carries its sample size and variance. Single-run point
  estimates are not results.
- B200 is a proxy for GB200 NVL72 — single GPU, no NVLink5 rack-scale
  interconnect.

## Out of scope

Speculative decoding acceptance on Malay, prefix caching against in-place
Mamba state, and quantization-aware distillation are stated hypotheses in the
report's further-work section, not experiments here.

## Layout

```
configs/         typed config + YAML
modal_app/       images, volumes, one function per phase
recipes/         forked ModelOpt recipes (both families)
src/ilmu_glossary/
  config.py      every constant in the study
  resolve.py     runtime HF id validation
  splits.py      80/20 splits + contamination guards
  data_prep.py   phase 0
  lid.py         Malay/Indonesian discrimination
  routing_analysis.py
  calib_select.py
  quantize.py
  evaluate/      kl, ppl, mmlu, throughput
  analyze.py
tests/
```
