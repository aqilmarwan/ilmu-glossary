"""Phase 1 - routing analysis.

Question: does expert routing differ by language, and by how much?

Method: forward passes over a stratified candidate pool capturing **router
logits only**. No quantization, no generation.

Two consumers depend on this phase's output and they need different things:

  * The report needs per-class aggregate statistics - activation frequency,
    entropy, pairwise Jaccard and KL, coverage curves.
  * Phase 2's coverage-greedy selector needs **per-document** expert
    activation profiles, because it scores each candidate document by the
    marginal coverage it would add.

Both are written. The per-document profiles dominate storage, so they are
kept as a sparse (doc, layer, expert, count) table rather than a dense
docs x layers x experts array, which at 50k docs would be tens of gigabytes.

Spec section 4's decision gate is evaluated here: if routing across
formal_bm, manglish, code_switched and english_control is statistically
indistinguishable, the central hypothesis is weakened. That is recorded and
the pipeline continues - corpus substitution is still worth testing, and the
report reframes around the null result rather than forcing the narrative.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ilmu_glossary import tracking
from ilmu_glossary.config import Config, CorpusClass
from ilmu_glossary.io import read_jsonl, write_json, write_parquet
from ilmu_glossary.seeds import rng, seed_everything

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Module names that identify a MoE router across architectures. Spec section 4
# warns that hybrid architectures place MoE layers irregularly and that layer
# indices must not be assumed, so discovery is by pattern over the actual
# module tree rather than by index arithmetic.
ROUTER_NAME_PATTERNS = (
    r"\.gate$",
    r"\.router$",
    r"\.router\.(gate|classifier|layer)$",
    r"\.block_sparse_moe\.gate$",
    r"\.mlp\.gate$",
    r"\.moe\.(gate|router)$",
    r"\.experts?\.(gate|router)$",
    r"\.gate_proj_router$",
)
_ROUTER_RE = re.compile("|".join(ROUTER_NAME_PATTERNS))

TOP_K_FOR_JACCARD = 32  # spec section 4


# --------------------------------------------------------------------------
# module discovery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterModule:
    """One discovered router, with the MoE layer index it belongs to."""

    name: str
    layer_index: int
    num_experts: int

    def __str__(self) -> str:
        return f"layer{self.layer_index}({self.name}, E={self.num_experts})"


def _layer_index_from_name(name: str) -> int:
    """Pull the transformer block index out of a dotted module path.

    Returns the *last* integer component, which is the block index for every
    convention seen in practice (`model.layers.17.mlp.gate`). Falls back to -1
    so an unparseable name still sorts deterministically rather than crashing.
    """
    parts = [p for p in name.split(".") if p.isdigit()]
    return int(parts[-1]) if parts else -1


def infer_num_experts(model: Any) -> int | None:
    """Read the routed-expert count from the model config, if it states one.

    Knowing the true count turns router discovery from a heuristic into an
    exact filter: on Nemotron-H the MoE router is `backbone.layers.N.mixer.gate`
    with out_features == n_routed_experts (128), while other `.gate`-suffixed
    projections in the same block have different widths.
    """
    config = getattr(model, "config", None)
    for attr in (
        "n_routed_experts",
        "num_experts",
        "moe_num_experts",
        "num_local_experts",
        "n_experts",
    ):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 1:
            return value
    return None


def discover_routers(model: Any) -> list[RouterModule]:
    """Locate every MoE router by walking the module tree.

    Spec section 9: "Router module names differ from standard MoE - inspect
    module tree; never assume layer indices."

    A router is identified as a Linear-like module whose name matches a router
    pattern and whose output dimension is plausibly an expert count. The
    dimension check is what prevents a `gate_proj` inside a dense MLP - which
    projects to the intermediate size, not the expert count - from being
    mistaken for a router.
    """
    expected = infer_num_experts(model)
    found: list[RouterModule] = []
    for name, module in model.named_modules():
        if not _ROUTER_RE.search(name):
            continue
        out_features = getattr(module, "out_features", None)
        if out_features is None:
            weight = getattr(module, "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            out_features = int(weight.shape[0])
        # An MoE router projects to num_experts, typically 8-512. A dense
        # gate_proj projects to the intermediate size, in the thousands.
        if not (2 <= int(out_features) <= 1024):
            continue
        found.append(
            RouterModule(
                name=name,
                layer_index=_layer_index_from_name(name),
                num_experts=int(out_features),
            )
        )

    if expected is not None:
        exact = [r for r in found if r.num_experts == expected]
        if exact:
            dropped = len(found) - len(exact)
            if dropped:
                logger.info(
                    "Discarded %d candidate router(s) whose width != n_routed_experts=%d",
                    dropped,
                    expected,
                )
            found = exact
        else:
            logger.warning(
                "No candidate router has out_features == n_routed_experts=%d. "
                "Keeping all %d candidates, but verify them against the module "
                "tree before trusting phase 1 - a wrong router set makes every "
                "routing statistic meaningless.",
                expected,
                len(found),
            )

    found.sort(key=lambda r: (r.layer_index, r.name))
    if not found:
        raise RuntimeError(
            "No router modules discovered. Inspect the module tree with "
            "`dump_module_tree(model)` and extend ROUTER_NAME_PATTERNS - "
            "do not fall back to assuming layer indices."
        )
    logger.info("Discovered %d routers: %s", len(found), ", ".join(str(r) for r in found[:8]))
    return found


def dump_module_tree(model: Any, *, max_lines: int = 400) -> str:
    """Human-readable module tree, for when discovery fails."""
    lines = []
    for name, module in model.named_modules():
        shape = getattr(module, "weight", None)
        dims = tuple(shape.shape) if shape is not None and hasattr(shape, "shape") else ()
        lines.append(f"{name}: {type(module).__name__} {dims}")
        if len(lines) >= max_lines:
            lines.append(f"... ({max_lines} shown)")
            break
    return "\n".join(lines)


def infer_top_k(model: Any, default: int = 8) -> int:
    """Read the router's top-k from the model config.

    Naming varies across architectures, so several conventions are tried
    before falling back. The fallback is logged loudly because a wrong top-k
    silently distorts every activation-frequency number in this phase.
    """
    config = getattr(model, "config", None)
    for attr in (
        "num_experts_per_tok",
        "moe_top_k",
        "top_k",
        "router_top_k",
        "num_selected_experts",
        "moe_router_topk",
    ):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    logger.warning(
        "Could not read top-k from model config; defaulting to %d. Verify this "
        "against the model card - a wrong top-k distorts every activation "
        "frequency in phase 1.",
        default,
    )
    return default


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


@dataclass
class RoutingCapture:
    """Accumulates expert assignment counts across a set of forward passes.

    `counts[layer]` is a length-num_experts int64 array of token assignments.
    Kept on CPU as numpy: 50k documents through ~30 MoE layers would otherwise
    hold a large amount of GPU memory hostage for no reason.
    """

    routers: list[RouterModule]
    top_k: int
    counts: dict[int, np.ndarray] = field(default_factory=dict)
    entropy_sums: dict[int, float] = field(default_factory=dict)
    token_totals: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for router in self.routers:
            self.counts.setdefault(router.layer_index, np.zeros(router.num_experts, dtype=np.int64))
            self.entropy_sums.setdefault(router.layer_index, 0.0)
            self.token_totals.setdefault(router.layer_index, 0)

    def observe(self, layer_index: int, logits: torch.Tensor) -> np.ndarray:
        """Record one layer's router logits for one forward pass.

        Returns the per-expert counts contributed by *this* call, which is
        what the per-document profile needs.
        """
        import torch

        flat = logits.reshape(-1, logits.shape[-1]).float()
        probs = torch.softmax(flat, dim=-1)

        k = min(self.top_k, flat.shape[-1])
        chosen = torch.topk(probs, k=k, dim=-1).indices.reshape(-1)
        local = torch.bincount(chosen.cpu(), minlength=self.counts[layer_index].shape[0])
        local_np = local.numpy().astype(np.int64)

        self.counts[layer_index] += local_np
        # Routing entropy over the full softmax, averaged per token. Low
        # entropy means confident routing; a language shifting entropy is
        # itself evidence even when the argmax set is unchanged.
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        self.entropy_sums[layer_index] += float(entropy.sum().item())
        self.token_totals[layer_index] += int(flat.shape[0])
        counted: np.ndarray = local_np
        return counted

    def frequency(self, layer_index: int) -> np.ndarray:
        """Normalised activation frequency vector for one layer."""
        counts = self.counts[layer_index]
        total = counts.sum()
        return counts / total if total else counts.astype(np.float64)

    def mean_entropy(self, layer_index: int) -> float:
        n = self.token_totals[layer_index]
        return self.entropy_sums[layer_index] / n if n else 0.0


class RouterProbe:
    """Registers forward hooks on router modules and captures their outputs.

    Spec section 4 permits `output_router_logits=True` where supported, but
    hooks are used unconditionally: the flag's semantics differ across
    architectures (some return pre-softmax logits, some post-softmax weights,
    some only for the last layer) and the hook path is identical everywhere.
    """

    def __init__(self, model: Any, routers: list[RouterModule], top_k: int) -> None:
        self.model = model
        self.routers = routers
        self.top_k = top_k
        self._handles: list[Any] = []
        self._buffer: dict[int, torch.Tensor] = {}

    def __enter__(self) -> RouterProbe:
        modules = dict(self.model.named_modules())
        for router in self.routers:
            module = modules[router.name]
            self._handles.append(module.register_forward_hook(self._make_hook(router.layer_index)))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._buffer.clear()

    def _make_hook(self, layer_index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # Routers sometimes return (logits, aux_loss) or a dataclass.
            tensor = output[0] if isinstance(output, tuple) else output
            if hasattr(tensor, "logits"):
                tensor = tensor.logits
            self._buffer[layer_index] = tensor.detach()

        return hook

    def drain(self) -> dict[int, torch.Tensor]:
        """Return and clear the logits captured by the last forward pass."""
        captured = dict(self._buffer)
        self._buffer.clear()
        return captured


# --------------------------------------------------------------------------
# candidate pool
# --------------------------------------------------------------------------


def build_candidate_pool(cfg: Config) -> list[dict[str, Any]]:
    """Stratified sample across the six classes (spec section 4: >=50,000 docs).

    Drawn from the **train** split only. Phase 2 selects calibration documents
    from these profiles, so a document appearing here that later turned up in
    evaluation would contaminate every variant at once.
    """
    from ilmu_glossary.splits import load_splits

    splits = load_splits(cfg.paths.resolve("splits"))
    stratified = cfg.paths.resolve("stratified")
    target_total = cfg.data.candidate_pool_size if not cfg.dry_run else 64

    available = {
        c.value: stratified / f"{c.value}.jsonl"
        for c in CorpusClass
        if (stratified / f"{c.value}.jsonl").exists()
    }
    if not available:
        raise FileNotFoundError(f"No stratified corpus files under {stratified}; run phase 0")

    per_class = max(target_total // len(available), 1)
    pool: list[dict[str, Any]] = []

    for class_name, path in available.items():
        train_indices = set(splits[class_name].train_indices)
        generator = rng(cfg.seed, "phase1", "pool", class_name)

        eligible = [i for i in sorted(train_indices)]
        take = min(per_class, len(eligible))
        chosen = set(
            generator.choice(np.array(eligible), size=take, replace=False).tolist() if take else []
        )

        for i, record in enumerate(read_jsonl(path)):
            if i in chosen:
                pool.append(
                    {
                        "doc_id": record.get("id", f"{class_name}:{i}"),
                        "corpus_class": class_name,
                        "doc_index": i,
                        # Calibration consumes the templated form, so route
                        # the templated form - routing on raw text would
                        # profile a distribution the PTQ pass never sees.
                        "text": record.get("templated") or record.get("text", ""),
                    }
                )

    logger.info("Candidate pool: %d documents across %d classes", len(pool), len(available))
    return pool


# --------------------------------------------------------------------------
# divergence statistics
# --------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-10) -> float:
    """KL(p || q) over two expert-frequency distributions.

    Smoothed because an expert that receives no tokens under q but some under
    p would otherwise send the divergence to infinity - and with sparse
    routing that happens constantly, which is precisely the phenomenon under
    study rather than a numerical accident to be hidden.
    """
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric, bounded alternative. Reported alongside KL because KL's
    asymmetry makes a table of pairwise values hard to read."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def top_k_jaccard(p: np.ndarray, q: np.ndarray, k: int = TOP_K_FOR_JACCARD) -> float:
    """Jaccard overlap of the top-k most-activated experts (spec section 4)."""
    k = min(k, len(p), len(q))
    if k == 0:
        return 0.0
    top_p = set(np.argsort(p)[-k:].tolist())
    top_q = set(np.argsort(q)[-k:].tolist())
    union = top_p | top_q
    return len(top_p & top_q) / len(union) if union else 0.0


def routing_significance(counts_a: np.ndarray, counts_b: np.ndarray) -> tuple[float, float, float]:
    """Chi-square test of independence between two classes' expert counts.

    Returns (chi2, p_value, cramers_v). Cramer's V is reported because with
    millions of routed tokens the chi-square p-value is essentially always
    below any threshold - effect size is what carries meaning here, and the
    spec's decision gate should turn on it rather than on significance alone.
    """
    from scipy import stats

    table = np.vstack([counts_a, counts_b]).astype(np.float64)
    # Drop experts unused by both classes; they contribute no information and
    # produce zero expected frequencies that invalidate the test.
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        return 0.0, 1.0, 0.0

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else 0.0
    return float(chi2), float(p_value), cramers_v


def coverage_curve(
    per_doc: np.ndarray, sample_counts: tuple[int, ...], *, seed: int
) -> list[dict[str, Any]]:
    """Expert coverage as a function of sample count (spec section 4).

    `per_doc` is (n_docs, n_experts) of per-document token counts. Reports the
    minimum and 10th-percentile per-expert token count at each N, which is
    exactly the quantity `coverage_greedy` maximises in phase 2 and the
    quantity that determines whether an expert's scales are estimable at all.
    """
    generator = np.random.default_rng(seed)
    n_docs = per_doc.shape[0]
    rows: list[dict[str, Any]] = []

    for n in sample_counts:
        if n > n_docs:
            continue
        # Several draws per N so the curve carries a variance, per spec
        # section 8's "single-run point estimates are not results".
        mins, p10s, zeros = [], [], []
        for trial in range(5):
            idx = generator.choice(n_docs, size=n, replace=False)
            totals = per_doc[idx].sum(axis=0)
            mins.append(float(totals.min()))
            p10s.append(float(np.percentile(totals, 10)))
            zeros.append(float((totals == 0).mean()))
            del trial
        rows.append(
            {
                "n_samples": n,
                "min_expert_tokens_mean": float(np.mean(mins)),
                "min_expert_tokens_std": float(np.std(mins, ddof=1)),
                "p10_expert_tokens_mean": float(np.mean(p10s)),
                "p10_expert_tokens_std": float(np.std(p10s, ddof=1)),
                "frac_experts_unseen_mean": float(np.mean(zeros)),
                "n_trials": 5,
            }
        )
    return rows


# --------------------------------------------------------------------------
# phase driver
# --------------------------------------------------------------------------


def _iter_batches(pool: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(pool), size):
        yield pool[start : start + size]


def run_phase1(cfg: Config) -> dict[str, Any]:
    """Capture routing over the candidate pool and compute divergence tables."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed_everything(cfg.seed, "phase1")
    fingerprint = cfg.fingerprint()
    routing_dir = cfg.paths.resolve("routing")
    results_dir = cfg.paths.resolve("results")

    with tracking.run(cfg, phase="phase1", run_name="routing_analysis"):
        repo = cfg.effective_model_repo()
        tokenizer = AutoTokenizer.from_pretrained(
            repo, trust_remote_code=cfg.model.trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=cfg.model.trust_remote_code,
        )
        model.eval()  # type: ignore[no-untyped-call]  # transformers is untyped

        routers = discover_routers(model)
        top_k = infer_top_k(model)
        num_experts = routers[0].num_experts
        logger.info("num_experts=%d top_k=%d moe_layers=%d", num_experts, top_k, len(routers))
        tracking.log_params(
            {"num_experts": num_experts, "top_k": top_k, "n_moe_layers": len(routers)}
        )

        pool = build_candidate_pool(cfg)
        max_len = min(cfg.effective_seq_len(), 8192)  # routing needs breadth, not length

        per_class: dict[str, RoutingCapture] = {}
        # Sparse per-document profiles for phase 2. Columns kept as flat lists
        # and assembled once, which is far cheaper than growing a DataFrame.
        profile_rows: dict[str, list[Any]] = {
            "doc_id": [],
            "corpus_class": [],
            "layer_index": [],
            "expert_id": [],
            "token_count": [],
        }

        with RouterProbe(model, routers, top_k) as probe, torch.inference_mode():
            for batch in _iter_batches(pool, size=1):
                doc = batch[0]
                encoded = tokenizer(
                    doc["text"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                encoded = {k: v.to(model.device) for k, v in encoded.items()}

                model(**encoded)
                captured = probe.drain()

                capture = per_class.setdefault(
                    doc["corpus_class"], RoutingCapture(routers=routers, top_k=top_k)
                )
                for layer_index, logits in captured.items():
                    local = capture.observe(layer_index, logits)
                    nonzero = np.nonzero(local)[0]
                    for expert_id in nonzero:
                        profile_rows["doc_id"].append(doc["doc_id"])
                        profile_rows["corpus_class"].append(doc["corpus_class"])
                        profile_rows["layer_index"].append(int(layer_index))
                        profile_rows["expert_id"].append(int(expert_id))
                        profile_rows["token_count"].append(int(local[expert_id]))

        # ------------------------------------------------ per-class tables
        for class_name, capture in per_class.items():
            for router in routers:
                layer = router.layer_index
                frequency = capture.frequency(layer)
                df = pd.DataFrame(
                    {
                        "expert_id": np.arange(len(frequency)),
                        "token_count": capture.counts[layer],
                        "activation_frequency": frequency,
                    }
                )
                df["corpus_class"] = class_name
                df["layer_index"] = layer
                df["mean_routing_entropy"] = capture.mean_entropy(layer)
                df["n_tokens"] = capture.token_totals[layer]
                write_parquet(
                    df,
                    routing_dir / f"{class_name}_layer{layer}.parquet",
                    fingerprint=fingerprint,
                    phase="phase1",
                )

        profiles = pd.DataFrame(profile_rows)
        write_parquet(
            profiles,
            routing_dir / "per_document_profiles.parquet",
            fingerprint=fingerprint,
            phase="phase1",
        )

        # ------------------------------------------------ pairwise divergence
        comparison = _pairwise_table(per_class, routers)
        write_parquet(
            comparison,
            results_dir / "routing_comparison.parquet",
            fingerprint=fingerprint,
            phase="phase1",
        )

        # ------------------------------------------------ coverage curves
        curves = _coverage_table(profiles, num_experts, cfg)
        write_parquet(
            curves,
            results_dir / "expert_coverage_curve.parquet",
            fingerprint=fingerprint,
            phase="phase1",
        )

        gate = evaluate_routing_gate(comparison)
        summary = {
            "num_experts": num_experts,
            "top_k": top_k,
            "n_moe_layers": len(routers),
            "router_names": [r.name for r in routers],
            "pool_size": len(pool),
            "classes": sorted(per_class),
            "routing_gate": gate,
            "config_fingerprint": fingerprint,
        }
        write_json(summary, results_dir / "phase1_summary.json")
        tracking.log_metrics(
            {
                "max_pairwise_cramers_v": gate["max_cramers_v"],
                "mean_top32_jaccard": gate["mean_jaccard"],
            }
        )
        logger.info("Routing gate: %s", gate["verdict"])

    return summary


def _pairwise_table(
    per_class: dict[str, RoutingCapture], routers: list[RouterModule]
) -> pd.DataFrame:
    """Pairwise KL, JS, Jaccard and significance, per layer per class pair."""
    rows: list[dict[str, Any]] = []
    for class_a, class_b in combinations(sorted(per_class), 2):
        cap_a, cap_b = per_class[class_a], per_class[class_b]
        for router in routers:
            layer = router.layer_index
            freq_a, freq_b = cap_a.frequency(layer), cap_b.frequency(layer)
            chi2, p_value, cramers_v = routing_significance(
                cap_a.counts[layer], cap_b.counts[layer]
            )
            rows.append(
                {
                    "class_a": class_a,
                    "class_b": class_b,
                    "layer_index": layer,
                    "kl_a_to_b": kl_divergence(freq_a, freq_b),
                    "kl_b_to_a": kl_divergence(freq_b, freq_a),
                    "jensen_shannon": jensen_shannon(freq_a, freq_b),
                    "top32_jaccard": top_k_jaccard(freq_a, freq_b),
                    "chi2": chi2,
                    "p_value": p_value,
                    "cramers_v": cramers_v,
                    "entropy_a": cap_a.mean_entropy(layer),
                    "entropy_b": cap_b.mean_entropy(layer),
                    "n_tokens_a": cap_a.token_totals[layer],
                    "n_tokens_b": cap_b.token_totals[layer],
                }
            )
    return pd.DataFrame(rows)


def _coverage_table(profiles: pd.DataFrame, num_experts: int, cfg: Config) -> pd.DataFrame:
    """Coverage curves per class, from the sparse per-document profiles."""
    rows: list[dict[str, Any]] = []
    sample_counts = cfg.effective_sample_counts()

    for class_name, group in profiles.groupby("corpus_class"):
        dense = _densify(group, num_experts)
        if dense.shape[0] == 0:
            continue
        for row in coverage_curve(dense, sample_counts, seed=cfg.seed):
            rows.append({"corpus_class": str(class_name), **row})
    return pd.DataFrame(rows)


def _densify(profiles: pd.DataFrame, num_experts: int) -> np.ndarray:
    """Sparse profile rows to a (n_docs, n_experts) matrix, summed over layers.

    Summing over layers is the right aggregation for the coverage question:
    an expert's scale estimate draws on tokens it saw anywhere in the network,
    and phase 2 selects whole documents rather than per-layer slices.
    """
    if profiles.empty:
        return np.zeros((0, num_experts), dtype=np.int64)

    doc_ids = profiles["doc_id"].unique()
    doc_to_row = {doc: i for i, doc in enumerate(doc_ids)}
    dense = np.zeros((len(doc_ids), num_experts), dtype=np.int64)

    rows = profiles["doc_id"].map(doc_to_row).to_numpy()
    cols = profiles["expert_id"].to_numpy()
    np.add.at(dense, (rows, cols), profiles["token_count"].to_numpy())
    return dense


def evaluate_routing_gate(
    comparison: pd.DataFrame,
    *,
    min_cramers_v: float = 0.05,
    max_jaccard: float = 0.95,
) -> dict[str, Any]:
    """Spec section 4 decision gate.

    Compares the language-bearing classes against `english_control`. If their
    routing is statistically indistinguishable, the central hypothesis is
    weakened - which is recorded, not hidden, and the pipeline continues.

    Thresholds are effect-size based. With millions of routed tokens a
    chi-square p-value is below any threshold regardless of whether the
    difference matters, so significance alone would open the gate every time.
    """
    language_classes = {
        CorpusClass.FORMAL_BM.value,
        CorpusClass.MANGLISH.value,
        CorpusClass.CODE_SWITCHED.value,
    }
    control = CorpusClass.ENGLISH_CONTROL.value

    relevant = comparison[
        (comparison["class_a"].isin(language_classes) & (comparison["class_b"] == control))
        | (comparison["class_b"].isin(language_classes) & (comparison["class_a"] == control))
    ]

    if relevant.empty:
        return {
            "verdict": "indeterminate",
            "reason": "no BM-vs-english_control comparisons available",
            "max_cramers_v": 0.0,
            "mean_jaccard": 0.0,
            "hypothesis_weakened": True,
        }

    max_v = float(relevant["cramers_v"].max())
    mean_jaccard = float(relevant["top32_jaccard"].mean())
    distinguishable = max_v >= min_cramers_v or mean_jaccard <= max_jaccard

    return {
        "verdict": "routing_differs" if distinguishable else "routing_indistinguishable",
        "reason": (
            f"max Cramer's V = {max_v:.4f} (threshold {min_cramers_v}), "
            f"mean top-{TOP_K_FOR_JACCARD} Jaccard = {mean_jaccard:.4f} "
            f"(threshold {max_jaccard})"
        ),
        "max_cramers_v": max_v,
        "mean_jaccard": mean_jaccard,
        "min_p_value": float(relevant["p_value"].min()),
        "n_comparisons": len(relevant),
        "hypothesis_weakened": not distinguishable,
        "note": (
            "Routing is statistically indistinguishable across languages. The "
            "central hypothesis is weakened. Corpus-substitution calibration "
            "remains worth testing, but the report must be framed around the "
            "null result rather than the routing narrative."
        )
        if not distinguishable
        else "",
    }


__all__ = [
    "ROUTER_NAME_PATTERNS",
    "TOP_K_FOR_JACCARD",
    "RouterModule",
    "RouterProbe",
    "RoutingCapture",
    "build_candidate_pool",
    "coverage_curve",
    "discover_routers",
    "dump_module_tree",
    "evaluate_routing_gate",
    "infer_num_experts",
    "infer_top_k",
    "jensen_shannon",
    "kl_divergence",
    "routing_significance",
    "run_phase1",
    "top_k_jaccard",
]
