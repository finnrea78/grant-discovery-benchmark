"""`uv run figures`: regenerate every table and figure from the frozen `eval/scored/grid/` data.

Writes `report.txt`, `results.json` (undefined values as `null`) and a set of PNG
figures. Any calculation a figure needs lives in a tested function, not in the drawing
code. Uses the `Agg` backend so it runs without a display.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from grant_eval.analysis import contrasts as con
from grant_eval.analysis import dataset as ds
from grant_eval.analysis import edgecases as ec
from grant_eval.analysis import fields as fld
from grant_eval.analysis import errors as err
from grant_eval.analysis import frontier as fr
from grant_eval.analysis import funders as fm
from grant_eval.analysis import generalisation as gen
from grant_eval.analysis import report
from grant_eval.analysis import sensitivity as sens
from grant_eval.analysis import stats as st
from grant_eval.analysis import telemetry as tel
from grant_eval.analysis import strata as strat
from grant_eval.schema import EdgeCaseTag
from grant_eval.scoring.groundtruth import load_ground_truth

__all__ = ["FiguresReport", "build_parser", "generate", "main"]

#: Where the artefacts land unless told otherwise.
DEFAULT_OUT_DIR = Path("eval/figures")

#: `recently_closed` only appears on distractors, so it is never an in-scope stratum.
_DISTRACTOR_ONLY_TAG = EdgeCaseTag.RECENTLY_CLOSED.value


@dataclass(frozen=True)
class FiguresReport:
    """What one `generate` call wrote, for the CLI to summarise."""

    out_dir: Path
    n_configs: int
    n_runs: int


def _config_row(config_dict: dict) -> dict:
    cfg = config_dict["config"]
    label = ds.config_label(cfg["model"], cfg["prompt_strategy"], cfg["temperature"])
    return {
        "config_label": label,
        "model": cfg["model"],
        "prompt_strategy": cfg["prompt_strategy"],
        "temperature": cfg["temperature"],
        "macro_f1": config_dict["macro_f1"],
        "macro_f1_undefined": config_dict["macro_f1_undefined"],
        "macro_precision": config_dict["macro_precision"],
        "macro_precision_undefined": config_dict["macro_precision_undefined"],
        "macro_recall": config_dict["macro_recall"],
        "n_funders": config_dict["n_funders"],
        "mean_stable_rate": config_dict["mean_stable_rate"],
        "mean_fleiss_kappa": config_dict["mean_fleiss_kappa"],
        "cost_total_usd": config_dict["cost"]["total_cost_usd"],
    }


def _per_funder_payload(configs: list[dict]) -> list[dict]:
    """One row per (config, funder) from `configs.json`, including undefined values.

    Most undefined precisions and kappas only show up at this level, not in the
    config-level macro figures.
    """
    rows = []
    for c in configs:
        cfg = c["config"]
        label = ds.config_label(cfg["model"], cfg["prompt_strategy"], cfg["temperature"])
        for fa in c["funders"]:
            funder = fa["funder"]
            cons = c["consistency"].get(funder) or {}
            rows.append(
                {
                    "config_label": label,
                    "funder": funder,
                    "precision": fa["precision"],
                    "precision_undefined": fa["precision_undefined"],
                    "recall": fa["recall"],
                    "f1": fa["f1"],
                    "stable_rate": cons.get("stable_rate"),
                    "stable_rate_undefined": cons.get("stable_rate_undefined"),
                    "fleiss_kappa": cons.get("fleiss_kappa"),
                    "fleiss_kappa_undefined": cons.get("fleiss_kappa_undefined"),
                }
            )
    return rows


def _funder_mean_f1(data: ds.AnalysisDataset) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for r in data.runs:
        sums[r.funder].append(r.f1)
    return {funder: sum(vals) / len(vals) for funder, vals in sums.items()}


def _strata_payload(data: ds.AnalysisDataset, gt) -> dict:
    policies = strat.stratum_policies(gt, data.funders())
    payload: dict[str, dict] = {}
    for tag in EdgeCaseTag:
        value = tag.value
        if value == _DISTRACTOR_ONLY_TAG:
            continue
        p = policies.get(value)
        if p is None:
            payload[value] = {
                "n_rows": 0,
                "n_clusters": 0,
                "inferential": False,
                "reason": "no tune rows carry this tag",
            }
        else:
            payload[value] = {
                "n_rows": p.n_rows,
                "n_clusters": p.n_clusters,
                "inferential": p.inferential,
                "reason": p.reason,
            }
    return payload


def _funder_type_payload(data: ds.AnalysisDataset, profiles: dict) -> dict:
    funder_mean_f1 = _funder_mean_f1(data)
    groups: dict[str, list[str]] = defaultdict(list)
    for name, profile in profiles.items():
        if profile.type_group is not None:
            groups[profile.type_group].append(name)
    payload = {}
    for group, names in groups.items():
        vals = [funder_mean_f1[n] for n in names if n in funder_mean_f1]
        payload[group] = {
            "n_funders": len(names),
            "mean_f1": (sum(vals) / len(vals)) if vals else None,
        }
    return payload


def _complexity_payload(data: ds.AnalysisDataset, profiles: dict) -> list[dict]:
    funder_mean_f1 = _funder_mean_f1(data)
    return [
        {
            "display_name": p.display_name,
            "slug": p.slug,
            "rows_per_url": p.rows_per_url,
            "band": fm.complexity_band(p),
            "mean_f1": funder_mean_f1.get(p.display_name),
        }
        for p in sorted(profiles.values(), key=lambda p: -p.rows_per_url)
    ]


def _build_payload(
    data: ds.AnalysisDataset,
    configs: list[dict],
    family,
    points,
    fp_profiles,
    fabrication,
    latency,
    abstention,
    cost_per_tp,
    edge_contrasts,
    edge_alone,
    tag_counts,
    tag_overlap,
    field_acc,
    field_by_config,
    field_by_prompt,
    field_by_temp,
    field_dirs,
    field_tags,
    e2e_deadlines,
    e2e_all,
    health,
    comparisons,
    profiles: dict,
    gt,
    failures,
    n_resamples: int,
    seed: int,
    generalisation: gen.GeneralisationResult | None = None,
) -> dict:
    """Every number this package computed, as JSON-ready data.

    `not_estimable` and `generalisation` are only added when they apply, so the tune
    output stays unchanged. `interaction` is `null` when the split lacks the levels it
    needs, with the reason under `not_estimable`.
    """
    manifest = data.manifest
    interaction_reasons = report._missing_interaction_levels(data)
    not_estimable = {}
    if interaction_reasons:
        not_estimable["interaction"] = interaction_reasons
    fixed = report.single_level_factors(data)
    if fixed:
        not_estimable["single_level_factors"] = fixed
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {
            "n_runs": manifest.get("counts", {}).get("n_scored", len(data.runs)),
            "n_configs": len(configs),
            "n_funders": len(data.funders()),
        },
        "provenance": {
            "scorer_source_hash": manifest["scorer"]["source_hash"],
            "manifest_created_utc": manifest.get("created_utc"),
        },
        "bootstrap": {
            "n_resamples": n_resamples,
            "seed": seed,
            "n_clusters": len(data.funders()),
        },
        "leaderboard": [_config_row(c) for c in configs],
        "per_funder": _per_funder_payload(configs),
        "contrasts": [asdict(c) for c in family],
        "strata": _strata_payload(data, gt),
        "funder_type": _funder_type_payload(data, profiles),
        "complexity": _complexity_payload(data, profiles),
        "frontier": [asdict(p) for p in points],
        "fp_reasons": [asdict(p) for p in fp_profiles],
        "fabrication": [asdict(p) for p in fabrication],
        "latency": [asdict(p) for p in latency],
        "abstention": [asdict(p) for p in abstention],
        "cost_per_true_positive": [asdict(r) for r in cost_per_tp],
        "edge_case_contrasts": [asdict(c) for c in edge_contrasts],
        "edge_case_contrasts_alone": [asdict(c) for c in edge_alone],
        "edge_case_counts": [asdict(b) for b in tag_counts],
        "edge_case_overlap": tag_overlap,
        "field_accuracy": [asdict(r) for r in field_acc],
        "field_accuracy_by_config": [asdict(r) for r in field_by_config],
        "field_accuracy_by_prompt": [asdict(r) for r in field_by_prompt],
        "field_accuracy_by_temperature": [asdict(r) for r in field_by_temp],
        "field_directions": [asdict(r) for r in field_dirs],
        "field_accuracy_by_tag": [asdict(r) for r in field_tags],
        "end_to_end_deadlines": [asdict(r) for r in e2e_deadlines],
        "end_to_end_all_fields": [asdict(r) for r in e2e_all],
        "runtime_health": asdict(health),
        "sensitivity": [asdict(c) for c in comparisons],
        "failures": asdict(failures),
        "interaction": None if interaction_reasons else asdict(
            con.interaction(
                data,
                "model", report._INTERACTION_MODEL_A, report._INTERACTION_MODEL_B,
                "prompt", "chain_of_thought", "minimal",
                predicted_direction="largest on nova-lite",
            )
        ),
    }
    if not_estimable:
        payload["not_estimable"] = not_estimable
    if generalisation is not None:
        payload["generalisation"] = asdict(generalisation)
    return payload


def _draw_f1_by_config(configs: list[dict], path: Path) -> None:
    """Horizontal bars, best first, coloured by model."""
    rows = sorted(configs, key=lambda c: -(c["macro_f1"] or 0.0))
    labels = [
        _short_config_label(
            ds.config_label(c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"])
        )
        for c in rows
    ]
    values = [c["macro_f1"] or 0.0 for c in rows]
    models = [c["config"]["model"] for c in rows]
    unique_models = sorted(set(models))
    palette = plt.get_cmap("tab10")
    color_of = {m: palette(i % 10) for i, m in enumerate(unique_models)}
    colors = [color_of[m] for m in models]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = list(range(len(rows)))[::-1]  # best config drawn at the top
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("macro F1")
    ax.set_title("F1 by config, best first, coloured by model")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_contrasts(family, path: Path) -> None:
    """Forest plot of the confirmatory contrasts with bootstrap CIs.

    Ten contrasts on the tune grid, one on the validation split.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [
        f"{c.factor}: {c.level_a.rsplit('/', 1)[-1]} vs {c.level_b.rsplit('/', 1)[-1]}"
        for c in family
    ]
    for i, c in enumerate(family):
        ci = c.difference_ci
        if ci.undefined:
            continue
        ax.plot([ci.lower, ci.upper], [i, i], color="gray", linewidth=1.5)
        ax.plot(ci.point, i, "o", color="crimson" if c.rejected else "black")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_yticks(range(len(family)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean difference (F1)")
    n_clusters = family[0].n_funders if family else 0
    ax.set_title(f"Confirmatory contrasts (D-A), 95% CI, n={n_clusters} clusters")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _short_config_label(config_label: str) -> str:
    """`config_label` without the vendor path, e.g. "nova-lite-v1 | minimal | t0.7".

    Only used for plot labels; tables and `results.json` keep the full label.
    """
    model, _, rest = config_label.partition(" | ")
    return f"{model.rsplit('/', 1)[-1]} | {rest}" if rest else model.rsplit("/", 1)[-1]


def _frontier_annotation(point) -> str:
    """Two-line label for a frontier point: config, then cost, F1 and P(on frontier).

    The probability is shown so that a config only just on the frontier does not look
    like a sure choice. It prints "n/a" if no resampling was done.
    """
    prob = point.frontier_probability
    prob_text = "n/a" if prob is None else f"{prob:.2f}"
    return (
        f"{_short_config_label(point.config_label)}\n"
        f"${point.cost_usd:.2f} · F1 {point.macro_f1:.3f} · P(front) {prob_text}"
    )


#: Label offsets for frontier points, cheapest first, placed in the empty space above
#: the curve. Tuned for three points and cycled if there are more.
_FRONTIER_LABEL_OFFSETS = ((10, 14, "left"), (-12, 26, "right"), (14, 10, "left"))


def _draw_frontier(points, path: Path) -> None:
    """Cost against macro F1, with the Pareto frontier joined and labelled.

    Points are coloured by model since the models form clear price clusters. Only
    frontier points are labelled, to keep the plot readable.
    """
    points = tuple(points)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    models = sorted({p.config_label.split(" | ", 1)[0] for p in points})
    palette = plt.get_cmap("tab10")
    color_of = {m: palette(i % 10) for i, m in enumerate(models)}
    for model in models:
        pts = [p for p in points if p.config_label.startswith(f"{model} | ")]
        ax.scatter(
            [p.cost_usd for p in pts],
            [p.macro_f1 for p in pts],
            color=color_of[model],
            s=38,
            alpha=0.85,
            label=model.rsplit("/", 1)[-1],
            zorder=2,
        )

    on = sorted((p for p in points if p.on_frontier), key=lambda p: p.cost_usd)
    ax.plot(
        [p.cost_usd for p in on],
        [p.macro_f1 for p in on],
        color="crimson",
        linewidth=1.4,
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgewidth=1.8,
        label="Pareto frontier",
        zorder=3,
    )

    for i, point in enumerate(on):
        dx, dy, ha = _FRONTIER_LABEL_OFFSETS[i % len(_FRONTIER_LABEL_OFFSETS)]
        ax.annotate(
            _frontier_annotation(point),
            xy=(point.cost_usd, point.macro_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=7.5,
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="crimson", alpha=0.9, linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color="crimson", linewidth=0.7, shrinkA=0, shrinkB=6),
            zorder=4,
        )

    ax.margins(x=0.12, y=0.14)
    ax.set_xlabel("total realised cost ($)")
    ax.set_ylabel("macro F1")
    ax.set_title("Cost-effectiveness frontier: total spend against macro F1")
    ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _mean_recall_by_model_and_tag(
    data: ds.AnalysisDataset, gt, tag: str, model: str
) -> float:
    """Mean recall for `model` on stratum `tag`, over all its (config, funder) cells.

    Kept out of the plotting loop so it can be tested.
    """
    model_runs = tuple(r for r in data.runs if r.model == model)
    by_config = strat.stratum_recall_by_funder(data, gt, tag, runs=model_runs)
    vals = [v for by_funder in by_config.values() for v in by_funder.values()]
    return sum(vals) / len(vals) if vals else 0.0



def _draw_fp_reasons(profiles, path: Path) -> None:
    """Each model's false positives as a share of its OWN total, stacked.

    Shares rather than counts, because the models have similar FP totals and what
    differs is the mix of reasons.
    """
    ordered = sorted(profiles, key=lambda p: p.label)
    labels = [p.label.rsplit("/", 1)[-1] for p in ordered]
    reasons = sorted(
        {r for p in ordered for r in p.by_reason},
        key=lambda r: -sum(p.by_reason.get(r, 0) for p in ordered),
    )
    palette = plt.get_cmap("tab20")
    color_of = {r: palette(i * 2 % 20) for i, r in enumerate(reasons)}

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y = list(range(len(ordered)))
    left = [0.0] * len(ordered)
    for reason in reasons:
        widths = [p.share.get(reason, 0.0) for p in ordered]
        ax.barh(y, widths, left=left, color=color_of[reason], label=reason, height=0.6)
        for i, (w, l) in enumerate(zip(widths, left)):
            if w >= 0.06:
                ax.text(l + w / 2, y[i], f"{w:.0%}", ha="center", va="center",
                        fontsize=7.5, color="black")
        left = [l + w for l, w in zip(left, widths)]

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{lab}\n(n={p.n_false_positives} FP)" for lab, p in zip(labels, ordered)],
        fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of that model's own false positives")
    ax.set_title("False-positive composition by model")
    ax.legend(fontsize=7.5, ncol=len(reasons), loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_latency(profiles, path: Path) -> None:
    """p50 bar with a whisker out to p95 and a marker at the max, on a log axis.

    Log scale because the models differ by two orders of magnitude. Percentiles are
    used instead of a mean because latency is heavily right-skewed.
    """
    ordered = sorted(profiles, key=lambda p: (p.p50_s is None, p.p50_s))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = list(range(len(ordered)))[::-1]

    for yi, p in zip(y, ordered):
        if p.p50_s is None:
            continue
        ax.barh(yi, p.p50_s, color="steelblue", height=0.45, label="_nolegend_", zorder=2)
        if p.p95_s is not None:
            ax.plot([p.p50_s, p.p95_s], [yi, yi], color="black", linewidth=1.2, zorder=3)
            ax.plot([p.p95_s], [yi], marker="|", color="black", markersize=10, zorder=3)
        if p.max_s is not None:
            ax.plot([p.max_s], [yi], marker="d", color="crimson", markersize=6, zorder=4)
        ax.text(p.p50_s, yi + 0.32, f"p50 {p.p50_s:,.0f}s", fontsize=7.5, va="bottom")

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([p.label.rsplit("/", 1)[-1] for p in ordered], fontsize=8)
    ax.set_xlabel("seconds per run (log scale) — bar = p50, whisker to p95, ♦ = max")
    ax.set_title("Latency by model: wall-clock seconds per run")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_abstention(profiles, path: Path) -> None:
    """Abstention rate by model. Abstaining gives zero recall for that funder."""
    ordered = sorted(profiles, key=lambda p: -(p.rate or 0.0))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [p.label.rsplit("/", 1)[-1] for p in ordered]
    rates = [p.rate or 0.0 for p in ordered]
    bars = ax.bar(labels, rates, color="darkorange", width=0.55)
    for bar, p in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"  {p.rate:.1%}\n  ({p.n_abstained}/{p.n_runs})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(rates + [0.01]) * 1.35)
    ax.set_ylabel("share of runs that declined to answer")
    ax.set_title("Abstention by model: runs that returned nothing")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_fabrication(profiles, path: Path) -> None:
    """Fabricated items by model, the most harmful kind of error for users."""
    ordered = sorted(profiles, key=lambda p: -p.total_fabricated)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [p.label.rsplit("/", 1)[-1] for p in ordered]
    totals = [p.total_fabricated for p in ordered]
    colors = ["crimson" if t else "seagreen" for t in totals]
    bars = ax.bar(labels, totals, color=colors, width=0.55)
    for bar, p in zip(bars, ordered):
        rate = "—" if p.mean_hallucination_rate is None else f"{p.mean_hallucination_rate:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"  {p.total_fabricated} items\n  in {p.n_runs_fabricating} runs\n  mean rate {rate}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(totals + [1]) * 1.5)
    ax.set_ylabel("fabricated items (total across 640 runs)")
    ax.set_title("Fabrication by model — invented items across each model's 640 runs")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_cost_per_tp(rows, path: Path) -> None:
    """Realised spend per recovered opportunity, cheapest first, coloured by model.

    Unlike the totals in `frontier.png`, this does not depend on how many runs each
    config had. Undefined costs (nothing recovered) are drawn as grey bars.
    """
    ordered = sorted(rows, key=lambda r: (r.cost_per_tp is None, r.cost_per_tp))
    models = sorted({r.label.split(" | ", 1)[0] for r in ordered})
    palette = plt.get_cmap("tab10")
    color_of = {m: palette(i % 10) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(10, 8))
    y = list(range(len(ordered)))[::-1]
    values = [(r.cost_per_tp if r.cost_per_tp is not None else 0.0) for r in ordered]
    colors = [
        "lightgray" if r.cost_per_tp is None else color_of[r.label.split(" | ", 1)[0]]
        for r in ordered
    ]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_config_label(r.label) for r in ordered], fontsize=6)
    ax.set_xlabel("realised cost per true positive ($) — grey = undefined, not zero")
    ax.set_title("Cost per recovered opportunity, by config")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def _draw_edge_cases(contrasts, path: Path) -> None:
    """Forest plot of each tag's recall against the untagged baseline.

    Markers are hollow because no contrast here is inferential (filled means a rejected
    null elsewhere). Each row shows its paired-cluster count.
    """
    ordered = sorted(contrasts, key=lambda c: -c.n_rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))

    undefined_rows = []
    for i, c in enumerate(ordered):
        ci = c.difference
        if ci is None or ci.undefined or ci.lower is None:
            undefined_rows.append(i)
            continue
        ax.plot([ci.lower, ci.upper], [i, i], color="gray", linewidth=1.5, zorder=2)
        ax.plot(ci.point, i, "o", color="white", markeredgecolor="black",
                markeredgewidth=1.2, markersize=7, zorder=3)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    # Axes coordinates, so the note is not read as a point estimate at zero.
    for i in undefined_rows:
        ax.text(0.015, i, "interval undefined — too few paired clusters",
                transform=ax.get_yaxis_transform(), fontsize=7.5,
                va="center", ha="left", color="gray", style="italic")

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(
        [f"{c.tag}\n{c.n_rows} row{'' if c.n_rows == 1 else 's'}, "
         f"{c.n_paired_clusters} paired cluster{'' if c.n_paired_clusters == 1 else 's'}"
         for c in ordered],
        fontsize=7.5,
    )
    ax.invert_yaxis()
    # Padding stops the bottom note overlapping the x tick labels.
    ax.set_ylim(len(ordered) - 0.4, -0.6)
    ax.set_xlabel("mean recall on tagged rows − mean recall on untagged rows")
    ax.set_title(
        "Do edge cases cost recall? Paired within funder — every interval crosses zero"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_edge_case_overlap(matrix, path: Path) -> None:
    """Heatmap of rows shared between tag pairs; the diagonal is each tag's own count.

    Shows how much the per-tag contrasts in `edge_cases.png` overlap.
    """
    tags = sorted(matrix)
    grid = [[matrix[a][b] for b in tags] for a in tags]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(grid, cmap="Blues")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags, fontsize=8)
    peak = max((v for row in grid for v in row), default=0)
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if peak and v > peak * 0.6 else "black")
    ax.set_title("Edge-case tag overlap (in-scope tune rows)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="rows carrying both tags")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def _draw_field_accuracy(rows, path: Path) -> None:
    """Grouped bars: each model's exact-match accuracy on each structured field.

    Each bar shows its evaluated count, since accuracy depends on detection and the
    counts differ a lot between models.
    """
    groups = sorted({r.label for r in rows})
    by = {(r.label, r.field): r for r in rows}
    palette = plt.get_cmap("tab10")
    color_of = {g: palette(i % 10) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(10, 5.2))
    n = max(len(groups), 1)
    width = 0.8 / n
    x = list(range(len(fld.FIELDS)))
    for gi, g in enumerate(groups):
        heights = [(by.get((g, f)).accuracy or 0.0) if by.get((g, f)) else 0.0 for f in fld.FIELDS]
        offsets = [xi + gi * width for xi in x]
        ax.bar(offsets, heights, width=width, color=color_of[g], label=g.rsplit("/", 1)[-1])
        for off, f in zip(offsets, fld.FIELDS):
            r = by.get((g, f))
            if r is None or r.accuracy is None:
                continue
            ax.text(off, r.accuracy, f"n={r.n_evaluated}", ha="center", va="bottom",
                    fontsize=6, rotation=90)

    ax.set_xticks([xi + width * n / 2 for xi in x])
    ax.set_xticklabels(fld.FIELDS, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("exact-match accuracy")
    ax.set_title("Structured field accuracy by model (conditional on detection — see n)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_field_by_prompt(rows, path: Path) -> None:
    """One panel per field: each model's accuracy under each prompt strategy.

    The y-axis is shared so that differences look the same size in every panel.
    """
    models = sorted({r.model for r in rows})
    levels = sorted({r.level for r in rows})
    by = {(r.model, r.level, r.field): r for r in rows}
    palette = plt.get_cmap("tab10")
    color_of = {m: palette(i % 10) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, len(fld.FIELDS), figsize=(15, 4.6), sharey=True)
    n = max(len(models), 1)
    width = 0.8 / n
    for ax, field in zip(axes, fld.FIELDS):
        x = list(range(len(levels)))
        for mi, model in enumerate(models):
            heights = [
                (by[(model, lv, field)].accuracy or 0.0) if (model, lv, field) in by else 0.0
                for lv in levels
            ]
            offsets = [xi + mi * width for xi in x]
            ax.bar(offsets, heights, width=width,
                   color=color_of[model], label=model.rsplit("/", 1)[-1])
            # Label real zeros so they are not mistaken for missing cells.
            for off, lv, h in zip(offsets, levels, heights):
                cell = by.get((model, lv, field))
                if cell is not None and cell.n_evaluated and h == 0.0:
                    ax.text(off, 0.012, "0", ha="center", va="bottom", fontsize=6,
                            color=color_of[model], fontweight="bold")
        ax.set_xticks([xi + width * n / 2 for xi in x])
        ax.set_xticklabels(levels, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(field, fontsize=10)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    axes[0].set_ylabel("exact-match accuracy")
    axes[0].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=len(models), loc="lower center",
               frameon=False)
    fig.suptitle("Field accuracy by prompt strategy, within model", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_end_to_end(rows, path: Path) -> None:
    """Detection rate against end-to-end rate per config, on one denominator.

    The gap between the two bars is what extraction errors cost on top of detection.
    """
    ordered = sorted(rows, key=lambda r: -(r.end_to_end_rate or 0.0))
    fig, ax = plt.subplots(figsize=(10, 8))
    y = list(range(len(ordered)))[::-1]
    ax.barh(y, [r.detection_rate or 0.0 for r in ordered], color="lightsteelblue",
            height=0.75, label="found", zorder=2)
    ax.barh(y, [r.end_to_end_rate or 0.0 for r in ordered], color="steelblue",
            height=0.45, label="found AND field correct", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_config_label(r.label) for r in ordered], fontsize=6)
    field = ordered[0].field if ordered else "?"
    ax.set_xlabel("share of all in-scope opportunities")
    ax.set_title(f"End-to-end result (field = {field}): the gap is what extraction costs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_field_by_tag(rows, path: Path) -> None:
    """Forest plot of field accuracy on tagged rows minus untagged rows.

    Hollow markers because none of these has enough paired clusters to be inferential.
    """
    ordered = [r for r in rows if r.difference is not None and not r.difference.undefined]
    ordered = sorted(ordered, key=lambda r: r.difference.point)
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.42 * len(ordered) + 1.6)))

    for i, r in enumerate(ordered):
        ci = r.difference
        colour = "crimson" if (ci.upper is not None and ci.upper < 0) else "gray"
        ax.plot([ci.lower, ci.upper], [i, i], color=colour, linewidth=1.5, zorder=2)
        ax.plot(ci.point, i, "o", color="white", markeredgecolor=colour,
                markeredgewidth=1.4, markersize=7, zorder=3)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(
        [f"{r.tag} × {r.field}  ({r.n_paired_clusters} paired)" for r in ordered],
        fontsize=7.5,
    )
    ax.set_ylim(-0.6, len(ordered) - 0.4)
    ax.set_xlabel("accuracy on tagged rows − accuracy on untagged rows")
    ax.set_title("Field accuracy by edge-case tag: red = interval entirely below zero")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_end_to_end_frontier(rows, path: Path) -> None:
    """Cost against the end-to-end rate, with the Pareto frontier labelled.

    Like `frontier.png`, but a missed opportunity and a misread one count the same.
    """
    ordered = [r for r in rows if r.end_to_end_rate is not None]
    models = sorted({r.label.split(" | ", 1)[0] for r in ordered})
    palette = plt.get_cmap("tab10")
    color_of = {m: palette(i % 10) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for m in models:
        pts = [r for r in ordered if r.label.startswith(f"{m} | ")]
        ax.scatter([r.cost_usd for r in pts], [r.end_to_end_rate for r in pts],
                   color=color_of[m], s=38, alpha=0.85, label=m.rsplit("/", 1)[-1], zorder=2)

    on = sorted((r for r in ordered if r.on_frontier), key=lambda r: r.cost_usd)
    ax.plot([r.cost_usd for r in on], [r.end_to_end_rate for r in on], color="crimson",
            linewidth=1.4, marker="o", markersize=9, markerfacecolor="none",
            markeredgewidth=1.8, label="Pareto frontier", zorder=3)

    for i, r in enumerate(on):
        dx, dy, ha = _FRONTIER_LABEL_OFFSETS[i % len(_FRONTIER_LABEL_OFFSETS)]
        # Escape "$": two of them in a label make matplotlib render mathtext.
        cpu = "—" if r.cost_per_usable is None else f"\\${r.cost_per_usable:.4f}"
        ax.annotate(
            f"{_short_config_label(r.label)}\n"
            f"\\${r.cost_usd:.2f} · {r.end_to_end_rate:.3f} · {cpu}/usable",
            xy=(r.cost_usd, r.end_to_end_rate), xytext=(dx, dy),
            textcoords="offset points", ha=ha, fontsize=7.5, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="crimson", alpha=0.9, linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color="crimson", linewidth=0.7,
                            shrinkA=0, shrinkB=6),
            zorder=4,
        )

    field = ordered[0].field if ordered else "?"
    ax.margins(x=0.12, y=0.16)
    ax.set_xlabel("total realised cost ($)")
    ax.set_ylabel("found AND correctly extracted")
    ax.set_title(f"Cost against the end-to-end result (field = {field})")
    ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_strata(data: ds.AnalysisDataset, gt, path: Path) -> None:
    """Mean recall by stratum, grouped by model."""
    policies = strat.stratum_policies(gt, data.funders())
    tags = sorted(policies.keys())
    models = sorted({r.model for r in data.runs})

    fig, ax = plt.subplots(figsize=(9, 6))
    n_models = max(len(models), 1)
    width = 0.8 / n_models
    x = list(range(len(tags)))
    for mi, model in enumerate(models):
        heights = [_mean_recall_by_model_and_tag(data, gt, tag, model) for tag in tags]
        offsets = [xi + mi * width for xi in x]
        ax.bar(offsets, heights, width=width, label=model.rsplit("/", 1)[-1])

    ax.set_xticks([xi + width * n_models / 2 for xi in x])
    ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("mean recall")
    ax.set_title("Recall by stratum and model (D-C)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_reliability(configs: list[dict], path: Path) -> None:
    """Mean Fleiss' kappa by config, with undefined values as grey bars."""
    rows = sorted(
        configs,
        key=lambda c: ds.config_label(c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"]),
    )
    labels = [
        _short_config_label(
            ds.config_label(c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"])
        )
        for c in rows
    ]
    values = [c["mean_fleiss_kappa"] for c in rows]
    xs = [v if v is not None else 0.0 for v in values]
    colors = ["lightgray" if v is None else "seagreen" for v in values]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = list(range(len(rows)))
    ax.barh(y, xs, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("mean Fleiss' kappa (grey = undefined, not zero)")
    ax.set_title("Reliability (RQ3): Fleiss' kappa by config")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_not_estimable(path: Path, title: str, reasons: list[str]) -> None:
    """Placeholder figure that states why a chart has no data.

    Keeps the expected filename in place without drawing empty axes that look like zeros.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5, 0.6, "not estimable on this split",
        ha="center", va="center", fontsize=13, color="crimson", transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.42, "\n".join(reasons),
        ha="center", va="top", fontsize=7, wrap=True, transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_interaction(data: ds.AnalysisDataset, path: Path) -> None:
    """Chain-of-thought minus minimal F1 per model; negative means the scaffold hurts.

    Uses `contrasts.factor_gap_by_level`. If the split lacks the needed levels, a
    "not estimable" panel is drawn instead.
    """
    reasons = report._missing_interaction_levels(data)
    if reasons:
        _draw_not_estimable(
            path,
            "CoT vs minimal gap per model (interaction D-B)",
            reasons,
        )
        return
    cells = con.cell_means(data, "model", "prompt")
    gaps_by_model = con.factor_gap_by_level(cells, "chain_of_thought", "minimal")
    models = sorted(gaps_by_model)
    gaps = [gaps_by_model[model] for model in models]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["crimson" if g < 0 else "seagreen" for g in gaps]
    ax.bar([m.rsplit("/", 1)[-1] for m in models], gaps, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("mean F1(chain_of_thought) - F1(minimal)")
    ax.set_title("CoT vs minimal gap per model (interaction D-B)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _generalisation_or_none(
    data: ds.AnalysisDataset, scored_dir: str | Path
) -> gen.GeneralisationResult | None:
    """The RQ6 tune-vs-validation comparison if the data is the validation split, else `None`.

    The split is read from the runs, not the directory name. Errors loading the
    chosen config are not caught.
    """
    if data.splits() != ("validation",):
        return None
    return gen.tune_validation_gap(validation_scored_dir=scored_dir)


def generate(
    out_dir: str | Path = DEFAULT_OUT_DIR,
    scored_dir: str | Path = ds.DEFAULT_SCORED_DIR,
    n_resamples: int = 2000,
    seed: int = st.DEFAULT_SEED,
) -> FiguresReport:
    """Write `report.txt`, `results.json` and all PNG figures to `out_dir`.

    `scored_dir` is passed to everything that reads the frozen files, so a validation
    run never mixes in tune data. Raises `FileNotFoundError` if there is no data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ds.load_dataset(scored_dir)
    gt = load_ground_truth()
    # Pass the split explicitly, as `load_profiles` would otherwise default to tune.
    splits = data.splits()
    profiles = fm.load_profiles(splits[0] if len(splits) == 1 else "tune")
    family = con.confirmatory_family(data)
    points = fr.cost_effectiveness(data, n_resamples=n_resamples, seed=seed)
    comparisons = sens.compare_with_and_without_failures(data)
    failures = sens.failure_breakdown(data, scored_dir=scored_dir)
    configs = json.loads((Path(scored_dir) / "configs.json").read_text())
    fp_profiles = err.fp_reason_totals(data)
    fabrication = err.fabrication_totals(data)
    latency = tel.latency_profiles(data)
    abstention = tel.abstention_profiles(data)
    cost_per_tp = tel.cost_per_true_positive(data)
    edge_contrasts = ec.tag_contrast_family(data, gt)
    edge_alone = ec.tag_contrast_family(data, gt, alone=True)
    tag_counts = ec.recall_by_tag_count(data, gt)
    tag_overlap = ec.tag_cooccurrence(gt, data.funders())
    field_acc = fld.field_accuracy(data, gt, scored_dir=scored_dir)
    field_dirs = fld.field_directions(data, scored_dir=scored_dir)
    field_by_config = fld.field_accuracy(data, gt, key="config", scored_dir=scored_dir)
    field_by_prompt = fld.field_accuracy_by_factor(data, gt, "prompt", scored_dir=scored_dir)
    field_by_temp = fld.field_accuracy_by_factor(data, gt, "temperature", scored_dir=scored_dir)
    e2e_deadlines = fld.end_to_end_frontier(
        fld.end_to_end(data, gt, field='deadlines', scored_dir=scored_dir)
    )
    e2e_all = fld.end_to_end_frontier(
        fld.end_to_end(data, gt, field='all', scored_dir=scored_dir)
    )
    field_tags = [
        fld.field_accuracy_by_tag(data, gt, f, t, scored_dir=scored_dir)
        for t in ('multi_deadline', 'restricted', 'rolling')
        for f in fld.FIELDS
    ]
    health = tel.runtime_health(scored_dir=scored_dir)
    generalisation = _generalisation_or_none(data, scored_dir)

    text = report.full_report(
        data, gt, profiles, family, points,
        scored_dir=scored_dir, generalisation=generalisation,
    )
    (out_dir / "report.txt").write_text(text)

    payload = _build_payload(
        data, configs, family, points, fp_profiles, fabrication, latency,
        abstention, cost_per_tp, edge_contrasts, edge_alone, tag_counts,
        tag_overlap, field_acc, field_by_config, field_by_prompt, field_by_temp,
        field_dirs, field_tags, e2e_deadlines,
        e2e_all, health, comparisons, profiles, gt, failures,
        n_resamples, seed, generalisation=generalisation,
    )
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str)
    )

    _draw_f1_by_config(configs, out_dir / "f1_by_config.png")
    _draw_contrasts(family, out_dir / "contrasts.png")
    _draw_frontier(points, out_dir / "frontier.png")
    _draw_strata(data, gt, out_dir / "strata.png")
    _draw_reliability(configs, out_dir / "reliability.png")
    _draw_interaction(data, out_dir / "interaction.png")
    _draw_fp_reasons(fp_profiles, out_dir / "fp_reasons.png")
    _draw_latency(latency, out_dir / "latency.png")
    _draw_abstention(abstention, out_dir / "abstention.png")
    _draw_fabrication(fabrication, out_dir / "fabrication.png")
    _draw_cost_per_tp(cost_per_tp, out_dir / "cost_per_tp.png")
    _draw_edge_cases(edge_contrasts, out_dir / "edge_cases.png")
    _draw_edge_case_overlap(tag_overlap, out_dir / "edge_case_overlap.png")
    _draw_field_accuracy(field_acc, out_dir / "field_accuracy.png")
    _draw_field_by_prompt(field_by_prompt, out_dir / "field_by_prompt.png")
    _draw_end_to_end(e2e_deadlines, out_dir / "end_to_end.png")
    _draw_field_by_tag(field_tags, out_dir / "field_by_tag.png")
    _draw_end_to_end_frontier(e2e_all, out_dir / "end_to_end_frontier.png")

    return FiguresReport(out_dir=out_dir, n_configs=len(configs), n_runs=len(data.runs))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="figures",
        description=(
            "Regenerate every table and figure in this package from the frozen scored "
            "grid: report.txt, results.json (every number, undefined as null) and the "
            "PNG figures. Runs headless (matplotlib Agg)."
        ),
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"default: {DEFAULT_OUT_DIR}")
    p.add_argument("--scored-dir", default=str(ds.DEFAULT_SCORED_DIR), help=f"default: {ds.DEFAULT_SCORED_DIR}")
    p.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples (default: 2000)")
    p.add_argument("--seed", type=int, default=st.DEFAULT_SEED, help=f"default: {st.DEFAULT_SEED}")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate(
            args.out_dir,
            scored_dir=args.scored_dir,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote figures for {result.n_configs} config(s), {result.n_runs} run(s) -> {result.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
