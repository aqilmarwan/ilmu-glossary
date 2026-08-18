"""Command line interface.

    uv run ilmu preflight          validate ids, config and recipes - free
    uv run ilmu config             print the resolved config and fingerprint
    uv run ilmu phase0 ... phase5  run one phase locally
    uv run ilmu report             regenerate REPORT.md from existing artifacts

Phases that need a GPU will fail on a laptop; that is intended. Use
`modal run modal_app/app.py` for the real runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from ilmu_glossary.config import Config, RecipeFamily

app = typer.Typer(
    add_completion=False,
    help="Routing-aware quantization calibration for Bahasa Melayu.",
    no_args_is_help=True,
)
console = Console()

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to config YAML")]
DryRunOption = Annotated[bool, typer.Option("--dry-run", help="Tiny model, tiny N")]

DEFAULT_CONFIG = Path("configs/config.yaml")


def _load(config: Path, dry_run: bool) -> Config:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return Config.load(config if config.exists() else None, dry_run=dry_run)


@app.command()
def config(
    config: ConfigOption = DEFAULT_CONFIG,
    dry_run: DryRunOption = False,
) -> None:
    """Print the resolved configuration and its fingerprint."""
    cfg = _load(config, dry_run)
    console.print(cfg.to_yaml())
    console.print(f"[bold]fingerprint:[/bold] {cfg.fingerprint()}")


@app.command()
def preflight(
    config: ConfigOption = DEFAULT_CONFIG,
    dry_run: DryRunOption = False,
    check_hub: Annotated[bool, typer.Option(help="Validate ids against HuggingFace")] = True,
) -> None:
    """Validate everything checkable before spending money.

    Spec section 1 warns that HuggingFace ids change and must be confirmed
    rather than trusted. This does that, plus the recipe identity check, so
    both failure modes surface in seconds instead of hours into a B200 run.
    """
    cfg = _load(config, dry_run)
    failures: list[str] = []

    table = Table(title="Preflight", show_lines=False)
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("config fingerprint", cfg.fingerprint())
    table.add_row("seed", str(cfg.seed))
    table.add_row("calibration seq len", str(cfg.effective_seq_len()))
    table.add_row("sample counts", str(cfg.effective_sample_counts()))

    # Recipes
    from ilmu_glossary.quantize import load_recipe

    for family in cfg.quant.families:
        try:
            recipe = load_recipe(cfg, family)
            table.add_row(
                f"recipe {family.value}",
                f"[green]{recipe.identity_hash}[/green] "
                f"algo={recipe.algorithm} act_bits={recipe.expert_activation_bits} "
                f"shipped={recipe.shipped_by_nvidia}",
            )
        except Exception as exc:
            failures.append(f"recipe {family.value}: {exc}")
            table.add_row(f"recipe {family.value}", f"[red]{exc}[/red]")

    # Two families must differ, or the mechanism contrast is vacuous.
    try:
        shipped = load_recipe(cfg, RecipeFamily.W4A16_SHIPPED)
        mechanism = load_recipe(cfg, RecipeFamily.W4A4_MECHANISM)
        if shipped.identity_hash == mechanism.identity_hash:
            failures.append("the two recipe families are identical")
            table.add_row("family contrast", "[red]families are identical[/red]")
        elif shipped.expert_activation_bits == mechanism.expert_activation_bits:
            failures.append("both families use the same expert activation width")
            table.add_row("family contrast", "[red]same expert activation width[/red]")
        else:
            table.add_row(
                "family contrast",
                f"[green]W{shipped.expert_activation_bits} vs "
                f"W{mechanism.expert_activation_bits} expert activations[/green]",
            )
    except Exception as exc:
        table.add_row("family contrast", f"[yellow]skipped: {exc}[/yellow]")

    if check_hub:
        from ilmu_glossary.resolve import assert_model_resolves, resolve_datasets

        try:
            resolutions = assert_model_resolves(
                cfg.model.bf16_repo,
                cfg.model.nvfp4_reference_repo,
                revision=cfg.model.revision,
            )
            for name, resolution in resolutions.items():
                table.add_row(f"model {name}", f"[green]{resolution.describe()}[/green]")
        except Exception as exc:
            failures.append(str(exc))
            table.add_row("models", f"[red]{exc}[/red]")

        dataset_ids = sorted(
            {spec.repo_id for specs in cfg.data.sources.values() for spec in specs}
        )
        for repo_id, resolution in resolve_datasets(dataset_ids).items():
            style = "green" if resolution.usable else "yellow"
            table.add_row(f"dataset {repo_id}", f"[{style}]{resolution.describe()}[/{style}]")

    console.print(table)

    if failures:
        console.print(f"\n[red bold]{len(failures)} blocking issue(s):[/red bold]")
        for failure in failures:
            console.print(f"  - {failure}")
        raise typer.Exit(code=1)
    console.print("\n[green bold]Preflight passed.[/green bold]")


@app.command()
def phase0(config: ConfigOption = DEFAULT_CONFIG, dry_run: DryRunOption = False) -> None:
    """Data acquisition and stratification."""
    from ilmu_glossary.data_prep import run_phase0

    console.print(run_phase0(_load(config, dry_run)))


@app.command()
def phase1(config: ConfigOption = DEFAULT_CONFIG, dry_run: DryRunOption = False) -> None:
    """Routing analysis (requires a GPU)."""
    from ilmu_glossary.routing_analysis import run_phase1

    console.print(run_phase1(_load(config, dry_run)))


@app.command()
def phase2(config: ConfigOption = DEFAULT_CONFIG, dry_run: DryRunOption = False) -> None:
    """Calibration set construction."""
    from ilmu_glossary.calib_select import run_phase2

    result = run_phase2(_load(config, dry_run))
    console.print(f"Built {result['n_sets']} calibration sets across {result['n_experts']} experts")


@app.command()
def phase3(
    config: ConfigOption = DEFAULT_CONFIG,
    dry_run: DryRunOption = False,
    variant: str = "baseline_en",
    sample_count: int = 1024,
    family: str = "w4a16_shipped",
) -> None:
    """One PTQ run (requires a GPU)."""
    from ilmu_glossary.quantize import run_single_quantization

    console.print(
        run_single_quantization(
            _load(config, dry_run), variant=variant, sample_count=sample_count, family=family
        )
    )


@app.command()
def phase4(
    config: ConfigOption = DEFAULT_CONFIG,
    dry_run: DryRunOption = False,
    checkpoint: str = "bf16_reference",
) -> None:
    """Evaluate one checkpoint (requires a GPU)."""
    from ilmu_glossary.evaluate import run_phase4

    console.print(run_phase4(_load(config, dry_run), checkpoint=checkpoint))


@app.command()
def phase5(config: ConfigOption = DEFAULT_CONFIG, dry_run: DryRunOption = False) -> None:
    """Analysis and REPORT.md."""
    from ilmu_glossary.analyze import run_phase5

    console.print(run_phase5(_load(config, dry_run)))


@app.command()
def report(config: ConfigOption = DEFAULT_CONFIG, dry_run: DryRunOption = False) -> None:
    """Regenerate REPORT.md from whatever artifacts exist."""
    from ilmu_glossary.analyze import run_phase5

    result = run_phase5(_load(config, dry_run))
    console.print(f"Wrote {result['report_path']} ({result['report_chars']} chars)")
    if result["missing_artifacts"]:
        console.print(f"[yellow]Missing: {', '.join(result['missing_artifacts'])}[/yellow]")


if __name__ == "__main__":
    app()
