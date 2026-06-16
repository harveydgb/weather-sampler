"""Batch runner for the 6-epoch forecast-lead Phase 4 audit.

This is deliberately only orchestration: it converts the forecast-format torch
artifact into per-lead .npz files, then invokes the existing Phase 4 real-data
runner once per lead with a lead-specific run directory.

Example smoke run:

    .venv/bin/python scripts/run_phase4_forecast_leads.py --dry-run --lead-steps 8

Full 6-epoch audit:

    .venv/bin/python scripts/run_phase4_forecast_leads.py

Use --skip-convert once the per-lead files already exist in outputs/data/.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = REPO_ROOT / "scripts" / "convert_real_gmm_pt_to_npz.py"
RUN_PHASE4_SCRIPT = REPO_ROOT / "scripts" / "run_phase4_real.py"
FIGURE_SCRIPT = REPO_ROOT / "scripts" / "make_phase4_figures.py"

DEFAULT_FORECAST_PT = Path("~/model_outputs/gmm_params_gmm_fc48_v1_me5_2t_f8.pt")
DEFAULT_PREFIX = "phase_4_fc48_6ep"
DEFAULT_DATA_DIR = REPO_ROOT / "outputs" / "data"
DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "runs"
DEFAULT_GRAPH_CACHE = REPO_ROOT / "outputs" / "runs" / "o96_knn_k8_graph.npz"
DEFAULT_FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

# Per-lead figures exclude `robustness`: that figure reads robustness_probes.json
# (written only by scripts/run_phase4_probes.py for the committed step-0 run dir),
# which the forecast leads never generate -> a bare `--figures` call would crash
# with FileNotFoundError. The faithfulness figure is produced separately by
# scripts/run_forecast_faithfulness.py.
LEAD_FIGURE_NAMES = ("maps", "pareto", "variogram", "spectrum", "enrichment")

LEAD_RE_TEMPLATE = r"^{prefix}_step(?P<step>[0-9]+)_2t\.npz$"


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]


def _path_arg(path: Path | str) -> str:
    return str(Path(path).expanduser())


def _default_convert_python() -> str:
    """Prefer the WeatherGenerator venv because conversion imports torch."""

    candidate = Path("~/WeatherGenerator/.venv/bin/python").expanduser()
    return str(candidate) if candidate.exists() else sys.executable


def parse_lead_steps(raw: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated forecast step numbers, e.g. ``1,8``."""

    if raw is None or raw.strip() == "":
        return None
    steps: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            step = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--lead-steps must be comma-separated integers, got {raw!r}"
            ) from exc
        if step <= 0:
            raise argparse.ArgumentTypeError("--lead-steps must be positive integers")
        steps.append(step)
    if not steps:
        return None
    return tuple(sorted(set(steps)))


def _lead_regex(prefix: str) -> re.Pattern[str]:
    return re.compile(LEAD_RE_TEMPLATE.format(prefix=re.escape(prefix)))


def discover_lead_files(
    data_dir: Path,
    prefix: str,
    lead_steps: Iterable[int] | None = None,
) -> list[tuple[int, Path]]:
    """Return existing ``{prefix}_step{k}_2t.npz`` files sorted numerically."""

    allowed = set(lead_steps) if lead_steps is not None else None
    regex = _lead_regex(prefix)
    found: list[tuple[int, Path]] = []
    for path in Path(data_dir).expanduser().glob(f"{prefix}_step*_2t.npz"):
        match = regex.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if allowed is None or step in allowed:
            found.append((step, path))
    return sorted(found, key=lambda item: item[0])


def expected_lead_files(
    data_dir: Path,
    prefix: str,
    lead_steps: Iterable[int],
) -> list[tuple[int, Path]]:
    """Expected per-lead files for a known step set, whether or not they exist."""

    return [
        (int(step), Path(data_dir).expanduser() / f"{prefix}_step{int(step)}_2t.npz")
        for step in sorted(set(lead_steps))
    ]


def infer_steps_from_forecast_pt(forecast_pt: Path) -> tuple[int, ...] | None:
    """Infer steps 1..N from a filename suffix such as ``_f8.pt``."""

    match = re.search(r"_f(?P<n>[0-9]+)\.pt$", Path(forecast_pt).name)
    if match is None:
        return None
    n_steps = int(match.group("n"))
    if n_steps <= 0:
        return None
    return tuple(range(1, n_steps + 1))


def conversion_command(
    *,
    convert_python: str,
    forecast_pt: Path,
    data_dir: Path,
    prefix: str,
) -> Command:
    return Command(
        "convert",
        (
            convert_python,
            str(CONVERT_SCRIPT),
            "--input",
            _path_arg(forecast_pt),
            "--out-dir",
            _path_arg(data_dir),
            "--format",
            "forecast",
            "--prefix",
            prefix,
        ),
    )


def lead_run_dir(runs_dir: Path, prefix: str, step: int) -> Path:
    return Path(runs_dir).expanduser() / f"{prefix}_step{step}"


def lead_figure_dir(figures_dir: Path, prefix: str, step: int) -> Path:
    return Path(figures_dir).expanduser() / f"{prefix}_step{step}"


def _phase4_base_command(
    *,
    phase4_python: str,
    data_npz: Path,
    run_dir: Path,
    graph_cache: Path,
    quick: bool,
) -> list[str]:
    argv = [
        phase4_python,
        str(RUN_PHASE4_SCRIPT),
        "--data",
        _path_arg(data_npz),
        "--out-dir",
        _path_arg(run_dir),
        "--graph-cache",
        _path_arg(graph_cache),
    ]
    if quick:
        argv.append("--quick")
    return argv


def build_lead_commands(
    *,
    lead_files: Sequence[tuple[int, Path]],
    prefix: str,
    runs_dir: Path,
    graph_cache: Path,
    quick: bool,
    figures: bool,
    figures_dir: Path,
    phase4_python: str,
) -> list[Command]:
    """Plan the per-lead Phase 4 commands, excluding the conversion command."""

    commands: list[Command] = []
    for step, data_npz in lead_files:
        run_dir = lead_run_dir(runs_dir, prefix, step)
        base = _phase4_base_command(
            phase4_python=phase4_python,
            data_npz=data_npz,
            run_dir=run_dir,
            graph_cache=graph_cache,
            quick=quick,
        )
        commands.append(Command(f"step{step}: phase4", tuple(base)))
        commands.append(Command(f"step{step}: lambda-star", tuple(base + ["--lambda-star"])))
        commands.append(Command(f"step{step}: scores", tuple(base + ["--stages", "scores"])))
        if figures:
            commands.append(
                Command(
                    f"step{step}: figures",
                    (
                        phase4_python,
                        str(FIGURE_SCRIPT),
                        "--only",
                        *LEAD_FIGURE_NAMES,
                        "--data",
                        _path_arg(data_npz),
                        "--out-dir",
                        _path_arg(run_dir),
                        "--fig-dir",
                        _path_arg(lead_figure_dir(figures_dir, prefix, step)),
                    ),
                )
            )
    return commands


def build_commands(
    *,
    forecast_pt: Path,
    prefix: str,
    data_dir: Path,
    runs_dir: Path,
    graph_cache: Path,
    lead_files: Sequence[tuple[int, Path]],
    skip_convert: bool,
    quick: bool,
    figures: bool,
    figures_dir: Path,
    convert_python: str,
    phase4_python: str,
) -> list[Command]:
    """Build a complete command list for dry-runs and tests."""

    commands: list[Command] = []
    if not skip_convert:
        commands.append(
            conversion_command(
                convert_python=convert_python,
                forecast_pt=forecast_pt,
                data_dir=data_dir,
                prefix=prefix,
            )
        )
    commands.extend(
        build_lead_commands(
            lead_files=lead_files,
            prefix=prefix,
            runs_dir=runs_dir,
            graph_cache=graph_cache,
            quick=quick,
            figures=figures,
            figures_dir=figures_dir,
            phase4_python=phase4_python,
        )
    )
    return commands


def _select_leads_for_dry_run(args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.skip_convert:
        return _select_existing_leads(args)
    existing = discover_lead_files(args.data_dir, args.prefix, args.lead_steps)
    if args.lead_steps is not None and not args.skip_convert:
        return expected_lead_files(args.data_dir, args.prefix, args.lead_steps)
    if existing:
        return existing
    inferred = infer_steps_from_forecast_pt(args.forecast_pt)
    if inferred is not None:
        print(
            f"[dry-run] no converted leads found; inferring steps "
            f"{inferred[0]}..{inferred[-1]} from {args.forecast_pt.name}"
        )
        return expected_lead_files(args.data_dir, args.prefix, inferred)
    raise SystemExit(
        f"no converted lead .npz files found in {args.data_dir} for prefix "
        f"{args.prefix!r}; pass --lead-steps for a dry-run before conversion"
    )


def _select_existing_leads(args: argparse.Namespace) -> list[tuple[int, Path]]:
    lead_files = discover_lead_files(args.data_dir, args.prefix, args.lead_steps)
    if args.lead_steps is not None:
        found_steps = {step for step, _ in lead_files}
        missing = [step for step in args.lead_steps if step not in found_steps]
        if missing:
            expected = ", ".join(
                str(path) for _, path in expected_lead_files(args.data_dir, args.prefix, missing)
            )
            raise SystemExit(
                f"missing requested converted lead file(s) for step(s) {missing}: {expected}"
            )
    if not lead_files:
        raise SystemExit(
            f"no converted lead .npz files found in {args.data_dir} for prefix {args.prefix!r}"
        )
    return lead_files


def run_command(command: Command) -> None:
    print(f"[{command.label}] {shlex.join(command.argv)}", flush=True)
    try:
        subprocess.run(command.argv, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"[{command.label}] executable not found: {command.argv[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"[{command.label}] failed with exit code {exc.returncode}: "
            f"{shlex.join(command.argv)}"
        ) from exc


def print_dry_run(commands: Sequence[Command]) -> None:
    for command in commands:
        print(f"[{command.label}] {shlex.join(command.argv)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-pt", type=Path, default=DEFAULT_FORECAST_PT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--lead-steps", type=parse_lead_steps, default=None,
                        help="comma-separated forecast steps, e.g. 1,8; default: all converted steps")
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--quick", action="store_true",
                        help="pass --quick through to run_phase4_real.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="print planned commands without executing them")
    parser.add_argument("--skip-convert", action="store_true",
                        help="use already-converted per-lead .npz files")
    parser.add_argument("--figures", action="store_true",
                        help="also run make_phase4_figures.py per lead")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR,
                        help="parent directory for per-lead figure directories")
    parser.add_argument("--convert-python", default=None,
                        help="Python executable for conversion (default: WeatherGenerator venv if present)")
    parser.add_argument("--phase4-python", default=sys.executable,
                        help="Python executable for Phase 4 scripts")
    args = parser.parse_args(argv)

    args.forecast_pt = args.forecast_pt.expanduser()
    args.data_dir = args.data_dir.expanduser()
    args.runs_dir = args.runs_dir.expanduser()
    args.graph_cache = args.graph_cache.expanduser()
    args.figures_dir = args.figures_dir.expanduser()
    args.convert_python = args.convert_python or _default_convert_python()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.dry_run:
        lead_files = _select_leads_for_dry_run(args)
        commands = build_commands(
            forecast_pt=args.forecast_pt,
            prefix=args.prefix,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            graph_cache=args.graph_cache,
            lead_files=lead_files,
            skip_convert=args.skip_convert,
            quick=args.quick,
            figures=args.figures,
            figures_dir=args.figures_dir,
            convert_python=args.convert_python,
            phase4_python=args.phase4_python,
        )
        print_dry_run(commands)
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_convert:
        if not args.forecast_pt.exists():
            raise SystemExit(
                f"forecast artifact not found: {args.forecast_pt}\n"
                "This runner does not launch WeatherGenerator inference; create the .pt "
                "separately, or pass --skip-convert if per-lead .npz files already exist."
            )
        run_command(
            conversion_command(
                convert_python=args.convert_python,
                forecast_pt=args.forecast_pt,
                data_dir=args.data_dir,
                prefix=args.prefix,
            )
        )

    lead_files = _select_existing_leads(args)
    commands = build_lead_commands(
        lead_files=lead_files,
        prefix=args.prefix,
        runs_dir=args.runs_dir,
        graph_cache=args.graph_cache,
        quick=args.quick,
        figures=args.figures,
        figures_dir=args.figures_dir,
        phase4_python=args.phase4_python,
    )
    for command in commands:
        run_command(command)


if __name__ == "__main__":
    main()
