"""Text tables for the full results breakdown.

Nothing here computes a statistic: each table formats values from the other analysis
modules or from the scored `configs.json`. Undefined values print as a dash, never 0.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from grant_eval.analysis import contrasts as con
from grant_eval.analysis import dataset as ds
from grant_eval.analysis import edgecases as ec
from grant_eval.analysis import fields as fld
from grant_eval.analysis import errors as err
from grant_eval.analysis import funders as fm
from grant_eval.analysis import generalisation as gen
from grant_eval.analysis import sensitivity as sens
from grant_eval.analysis import strata as st
from grant_eval.analysis import telemetry as tel
from grant_eval.schema import EdgeCaseTag
from grant_eval.scoring.groundtruth import load_ground_truth

__all__ = [
    "leaderboard",
    "per_funder_table",
    "contrast_table",
    "interaction_note",
    "temperature_note",
    "stratum_table",
    "funder_type_table",
    "complexity_table",
    "frontier_table",
    "reliability_table",
    "failure_summary",
    "sensitivity_table",
    "edge_case_table",
    "field_accuracy_table",
    "field_direction_table",
    "field_by_tag_table",
    "field_by_config_table",
    "field_factor_table",
    "end_to_end_table",
    "runtime_health_table",
    "tag_count_table",
    "tag_overlap_table",
    "fp_reason_table",
    "fabrication_table",
    "latency_table",
    "abstention_table",
    "cost_per_tp_table",
    "generalisation_note",
    "single_level_factors",
    "scope_notes",
    "full_report",
]

#: What undefined values print as, matching `aggregate._fmt`.
DASH = "—"

#: `recently_closed` only ever tags distractors, so it is never an in-scope stratum.
#: Every other `EdgeCaseTag` is reported, even when absent.
_DISTRACTOR_ONLY_TAG = EdgeCaseTag.RECENTLY_CLOSED.value


def _fmt(value: float | None, decimals: int = 3) -> str:
    """Format a number, or a dash for `None`."""
    return DASH if value is None else f"{value:.{decimals}f}"


def _col_width(header: str, values) -> int:
    """Column width: the widest of the header and the values."""
    return max(len(header), max((len(v) for v in values), default=0))


def _ljust(value: str, width: int, gutter: int = 2) -> str:
    """Left-justify `value` and add a gutter, so fields never run together even when
    `value` is wider than `width`."""
    return value.ljust(width) + " " * gutter


def _short_level(factor: str, level: str) -> str:
    """Short display form of a factor level: model IDs lose their `openrouter/<vendor>/`
    prefix. `contrast_table` prints a legend with the full names."""
    if factor == "model":
        return level.rsplit("/", 1)[-1]
    return level


#: Printed in place of a table when a section can't be estimated on a split. The
#: validation split fixes model and temperature, so some sections have nothing to show,
#: and keeping the heading keeps the tune and validation outputs aligned.
_NOT_ESTIMABLE = "not estimable on this split: "


def single_level_factors(data: ds.AnalysisDataset) -> dict[str, str]:
    """`{factor: its only level}` for every factor with a single level in `data`.

    These factors have no within-factor contrast. Public because the notebooks pass it
    to `contrast_table`.
    """
    return {
        factor: levels[0]
        for factor, levels in con.factor_levels(data).items()
        if len(levels) == 1
    }


def _split_name(data: ds.AnalysisDataset) -> str:
    """The split name read from the runs, joined with `+` if mixed."""
    splits = data.splits()
    return splits[0] if len(splits) == 1 else "+".join(splits)


def _not_estimable_note(heading: str, reasons: list[str]) -> str:
    """The `heading` with one `_NOT_ESTIMABLE` line per reason, in place of a table."""
    return "\n".join([heading] + [f"  {_NOT_ESTIMABLE}{reason}" for reason in reasons])


def _load_configs(scored_dir: Path = ds.DEFAULT_SCORED_DIR) -> list[dict]:
    """`configs.json` from the scored directory, as `freeze_scores.py` wrote it."""
    return json.loads((Path(scored_dir) / "configs.json").read_text())


def _find_config(configs: list[dict], model: str, prompt: str, temperature: float) -> dict:
    """The `configs.json` entry for `(model, prompt, temperature)`, ignoring
    `agent_config`."""
    for c in configs:
        cfg = c["config"]
        if (
            cfg["model"] == model
            and cfg["prompt_strategy"] == prompt
            and float(cfg["temperature"]) == float(temperature)
        ):
            return c
    raise KeyError(f"no config for {model!r} | {prompt!r} | t{temperature}")


def _parse_label(label: str) -> tuple[str, str, float]:
    """Inverts `dataset.config_label`: `"<model> | <prompt> | t<temperature>"`."""
    model, prompt, temp = label.split(" | ")
    return model, prompt, float(temp[1:])


def _funder_mean_f1(data: ds.AnalysisDataset) -> dict[str, float]:
    """Each funder's mean F1 over every run in the grid, across all configs."""
    sums: dict[str, list[float]] = defaultdict(list)
    for r in data.runs:
        sums[r.funder].append(r.f1)
    return {funder: sum(vals) / len(vals) for funder, vals in sums.items()}


def leaderboard(data: ds.AnalysisDataset, scored_dir: Path = ds.DEFAULT_SCORED_DIR) -> str:
    """Every config, best macro F1 first, from `configs.json`."""
    configs = _load_configs(scored_dir)

    def sort_key(c: dict) -> tuple[bool, float]:
        f1 = c["macro_f1"]
        return (f1 is None, -(f1 if f1 is not None else 0.0))

    ordered = sorted(configs, key=sort_key)
    labels = [
        ds.config_label(c["config"]["model"], c["config"]["prompt_strategy"], c["config"]["temperature"])
        for c in ordered
    ]
    config_w = _col_width("config", labels)

    out = [f"LEADERBOARD — macro F1 across all {len(ordered)} configurations, best first"]
    out.append(
        f"  {'#':>3}  {_ljust('config', config_w)}{'macroF1':>9}{'macroP':>9}{'macroR':>9}{'funders':>9}"
    )
    for i, (c, label) in enumerate(zip(ordered, labels), start=1):
        out.append(
            f"  {i:>3}  {_ljust(label, config_w)}"
            f"{_fmt(c['macro_f1']):>9}"
            f"{_fmt(c['macro_precision']):>9}"
            f"{_fmt(c['macro_recall']):>9}"
            f"{c['n_funders']:>9}"
        )
    return "\n".join(out)


def per_funder_table(
    data: ds.AnalysisDataset,
    config_label: str,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
) -> str:
    """Per-funder precision, recall, F1, stability and Fleiss' kappa for one config.

    Kappa is undefined under complete agreement and prints as a dash.
    """
    model, prompt, temperature = _parse_label(config_label)
    configs = _load_configs(scored_dir)
    c = _find_config(configs, model, prompt, temperature)

    funder_names = [fa["funder"] for fa in c["funders"]]
    funder_w = _col_width("funder", funder_names)

    out = [f"PER-FUNDER — {config_label}"]
    out.append(
        f"  {_ljust('funder', funder_w)}{'n':>4}{'reps':>5}{'prec':>7}{'rec':>7}{'F1':>7}"
        f"{'stable':>8}{'kappa':>8}"
    )
    for fa in c["funders"]:
        funder = fa["funder"]
        cons = c["consistency"].get(funder) or {}
        out.append(
            f"  {_ljust(funder, funder_w)}{fa['n_in_scope']:>4}{fa['n_replicates']:>5}"
            f"{_fmt(fa['precision']):>7}{fa['recall']:>7.3f}{fa['f1']:>7.3f}"
            f"{_fmt(cons.get('stable_rate')):>8}"
            f"{_fmt(cons.get('fleiss_kappa')):>8}"
        )
    out.append("")
    if c["excluded_funders"]:
        out.append(
            "excluded (zero-positive) funders, scored on abstention only: "
            + ", ".join(c["excluded_funders"])
        )
    else:
        out.append("no zero-positive funders excluded")
    return "\n".join(out)


def contrast_table(family, fixed_factors: dict[str, str] | None = None) -> str:
    """The pre-registered within-factor contrasts, with effect sizes and CIs.

    The CI header shows the number of funder clusters. `fixed_factors` (from
    `single_level_factors`) lists factors that get a not-estimable line instead of rows.
    """
    n_clusters = family[0].n_funders if family else 0
    ci_header = f"95% CI (n={n_clusters} clusters)"
    ci_w = max(len(ci_header) + 2, 26)

    displayed = [
        (c, _short_level(c.factor, c.level_a), _short_level(c.factor, c.level_b))
        for c in family
    ]
    factor_w = _col_width("factor", (c.factor for c in family))
    level_a_w = _col_width("level_a", (a for _, a, _ in displayed))
    level_b_w = _col_width("level_b", (b for _, _, b in displayed))

    out = ["CONFIRMATORY CONTRASTS (D-A) — pre-registered within-factor family, Holm-corrected together"]
    out.append(
        f"  {_ljust('factor', factor_w)}{_ljust('level_a', level_a_w)}{_ljust('level_b', level_b_w)}"
        f"{'mean d':>9}{'median d':>10}{'rank-biserial':>15}{'p':>9}{'Holm p':>9}"
        f"{ci_header:>{ci_w}}{'reject':>8}"
    )
    for c, level_a, level_b in displayed:
        ci = c.difference_ci
        ci_str = DASH if ci.undefined else f"[{ci.lower:.3f}, {ci.upper:.3f}]"
        reject = DASH if c.rejected is None else ("yes" if c.rejected else "no")
        out.append(
            f"  {_ljust(c.factor, factor_w)}{_ljust(level_a, level_a_w)}{_ljust(level_b, level_b_w)}"
            f"{c.mean_difference:>9.3f}{_fmt(c.median_difference):>10}"
            f"{_fmt(c.wilcoxon.rank_biserial):>15}{_fmt(c.p_value):>9}{_fmt(c.adjusted_p):>9}"
            f"{ci_str:>{ci_w}}{reject:>8}"
        )

    legend = {}
    for c in family:
        if c.factor == "model":
            legend[_short_level("model", c.level_a)] = c.level_a
            legend[_short_level("model", c.level_b)] = c.level_b
    if legend:
        out.append("")
        out.append("model identifiers (shortened above; full identifier for provenance):")
        for short, full in sorted(legend.items()):
            out.append(f"  {short} = {full}")

    for factor, level in sorted((fixed_factors or {}).items()):
        out.append(
            f"  {_NOT_ESTIMABLE}factor {factor!r} has a single level ({level}), so it "
            "contributes no within-factor contrast"
        )

    return "\n".join(out)


#: The pre-registered chain_of_thought x model interaction, expected to be largest on
#: nova-lite because it has no reasoning mode of its own.
_INTERACTION_MODEL_A = "openrouter/amazon/nova-lite-v1"
_INTERACTION_MODEL_B = "openrouter/deepseek/deepseek-v4-flash-0731"


def _missing_interaction_levels(data: ds.AnalysisDataset) -> list[str]:
    """One reason string per level `interaction_note` needs that the split lacks."""
    levels = con.factor_levels(data)
    wanted = (
        ("model", _INTERACTION_MODEL_A),
        ("model", _INTERACTION_MODEL_B),
        ("prompt", "chain_of_thought"),
        ("prompt", "minimal"),
    )
    reasons = []
    for factor, level in wanted:
        present = levels.get(factor, ())
        if level not in present:
            reasons.append(
                f"factor {factor!r} has no level {level!r} (present: "
                + ", ".join(present)
                + ")"
            )
    return reasons


def interaction_note(data: ds.AnalysisDataset) -> str:
    """The pre-registered interaction contrast, stated as a number and a direction.

    It is a separate one-comparison family from `contrast_table`. If the split lacks
    any of the four levels it needs (as on validation), a not-estimable note is
    returned instead.
    """
    missing = _missing_interaction_levels(data)
    if missing:
        return _not_estimable_note("PRE-REGISTERED INTERACTION (D-B)", missing)
    result = con.interaction(
        data,
        "model", _INTERACTION_MODEL_A, _INTERACTION_MODEL_B,
        "prompt", "chain_of_thought", "minimal",
        predicted_direction="largest on nova-lite",
    )
    ci = result.ci
    ci_str = DASH if ci.undefined else f"[{ci.lower:.3f}, {ci.upper:.3f}]"
    did = result.difference_in_differences
    direction = "negative" if did < 0 else ("positive" if did > 0 else "zero")
    model_a = _INTERACTION_MODEL_A.rsplit("/", 1)[-1]
    model_b = _INTERACTION_MODEL_B.rsplit("/", 1)[-1]

    if direction == "negative":
        closing = (
            f"The interaction is {direction}: the scaffold pre-registration predicted "
            "would help most on nova-lite — it has no reasoning mode of its own — "
            "instead hurts most there, the opposite of the predicted direction."
        )
    elif direction == "positive":
        closing = (
            f"The interaction is {direction}, matching the predicted direction: the "
            "scaffold helps most exactly where nova-lite's lack of a reasoning mode "
            "predicted it would help most."
        )
    else:
        closing = "The interaction is exactly zero on this data."

    out = [
        "PRE-REGISTERED INTERACTION (D-B)",
        f"Predicted: {result.predicted_direction}. Observed: {result.observed_direction}.",
        (
            f"(chain_of_thought - minimal) on {model_a} minus the same gap on {model_b} "
            f"= {did:.3f} ({direction}), 95% CI {ci_str} (n={result.n_funders} clusters)."
        ),
        closing,
    ]
    if result.excluded:
        out.append(
            "excluded (undefined in at least one of the four cells): "
            + ", ".join(f"{name} ({reason})" for name, reason in result.excluded)
        )
    return "\n".join(out)


def temperature_note(data: ds.AnalysisDataset, family) -> str:
    """The temperature main effect and the temperature x model interaction for each
    model pair.

    A bare "no main effect" could hide an interaction, so both are shown. Needs at
    least two temperatures and two models, otherwise a not-estimable note is returned.
    """
    levels = con.factor_levels(data)
    reasons = [
        f"factor {factor!r} has a single level ({levels[factor][0]}), so there is no "
        f"{'temperature contrast to correct' if factor == 'temperature' else 'model pair to interact it with'}"
        for factor in ("temperature", "model")
        if len(levels.get(factor, ())) < 2
    ]
    if reasons:
        return _not_estimable_note(
            "TEMPERATURE: MAIN EFFECT AND INTERACTION WITH MODEL", reasons
        )

    main = next((c for c in family if c.factor == "temperature"), None)
    main_p = _fmt(main.adjusted_p) if main is not None else DASH

    models = con.factor_levels(data)["model"]
    out = [
        "TEMPERATURE: MAIN EFFECT AND INTERACTION WITH MODEL",
        f"No temperature main effect (Holm p {main_p}; see CONFIRMATORY CONTRASTS above).",
        "Temperature x model interaction (t0.7 - t0.0 difference-in-differences), "
        "funder-clustered bootstrap CI:",
    ]
    any_undefined = False
    all_cross = True
    for model_a, model_b in itertools.combinations(models, 2):
        result = con.interaction(data, "model", model_a, model_b, "temperature", "0.7", "0.0")
        ci = result.ci
        short_a = _short_level("model", model_a)
        short_b = _short_level("model", model_b)
        if ci.undefined:
            any_undefined = True
            out.append(f"  {short_a} vs {short_b}: undefined ({ci.reason})")
            continue
        crosses = ci.lower <= 0.0 <= ci.upper
        all_cross = all_cross and crosses
        out.append(
            f"  {short_a} vs {short_b}: {result.difference_in_differences:+.4f} "
            f"[{ci.lower:+.3f}, {ci.upper:+.3f}] (n={result.n_funders} clusters) — "
            f"{'crosses zero' if crosses else 'does not cross zero'}"
        )

    if any_undefined:
        out.append(
            "at least one interaction is undefined (see above) — the conclusion "
            "below covers only the pairs actually computed."
        )
    if all_cross:
        out.append(
            "Every computed interval crosses zero: there is no evidence of a "
            "temperature x model interaction in F1. Combined with the main effect "
            "above, there is no detectable effect of temperature on F1, overall or "
            "for any model. Two caveats on that: these three interaction contrasts "
            "are POST HOC (examined only after the main-effect result was seen) and "
            "UNCORRECTED for multiplicity, unlike the ten-contrast confirmatory "
            "family above, so 'no detectable effect' is as far as this goes — it is "
            "not evidence of no effect. And this conclusion is scoped to F1 only: it "
            "says nothing about temperature's effect on run-to-run RELIABILITY, "
            "which is reported separately below and is clearly affected by "
            "temperature for some configs."
        )
    else:
        out.append(
            "At least one interval does not cross zero: unlike the main effect, this "
            "is evidence of a temperature x model interaction and should not be "
            "waved away as 'no temperature effect'."
        )
    return "\n".join(out)


def stratum_table(data: ds.AnalysisDataset, gt) -> str:
    """Every edge-case tag with its row and funder counts on this split, and whether
    it can be tested. Tags with no rows are listed as absent.
    """
    funders = data.funders()
    policies = st.stratum_policies(gt, funders)
    split = _split_name(data)

    out = [f"STRATA (D-C) — edge-case tags, {split}-only row/cluster counts"]
    out.append(f"  {'tag':<24}{'rows':>6}{'clusters':>10}{'status':>14}  reason")
    for tag in EdgeCaseTag:
        value = tag.value
        if value == _DISTRACTOR_ONLY_TAG:
            continue
        p = policies.get(value)
        if p is None:
            out.append(
                f"  {value:<24}{0:>6}{0:>10}{'absent':>14}  no {split} rows carry this tag"
            )
            continue
        status = "inferential" if p.inferential else "descriptive"
        reason = p.reason or "clears both D-C thresholds"
        out.append(f"  {p.tag:<24}{p.n_rows:>6}{p.n_clusters:>10}{status:>14}  {reason}")
    return "\n".join(out)


def funder_type_table(data: ds.AnalysisDataset, profiles: dict) -> str:
    """Mean F1 by funder type group. Exploratory: the groups were defined after seeing
    the tune data and are not tested."""
    funder_mean_f1 = _funder_mean_f1(data)
    groups: dict[str, list[str]] = defaultdict(list)
    for name, profile in profiles.items():
        if profile.type_group is None:
            continue
        groups[profile.type_group].append(name)

    out = [
        "FUNDER TYPE (D-F) — EXPLORATORY, defined after seeing the tune data; "
        "never statistically tested"
    ]
    out.append(f"  {'type_group':<18}{'n_funders':>10}{'mean F1 (all runs)':>21}")
    for group in sorted(groups):
        names = groups[group]
        vals = [funder_mean_f1[n] for n in names if n in funder_mean_f1]
        mean_v = sum(vals) / len(vals) if vals else None
        out.append(f"  {group:<18}{len(names):>10}{_fmt(mean_v):>21}")
    if not groups:
        out.append(
            f"  {_NOT_ESTIMABLE}no funder on this split carries a type group. "
            "`funders.TYPE_GROUPS` is validated to cover the tune split only "
            "(funders.load_profiles), deliberately: D-F was defined after seeing the "
            "tune data and bucketing a held-out funder into a post-hoc group would "
            "make an exploratory grouping look like a pre-registered one."
        )
    return "\n".join(out)


def complexity_table(data: ds.AnalysisDataset, profiles: dict) -> str:
    """`rows_per_url` and mean F1 per funder. Exploratory.

    Partly a scoring effect: titles are only checked on shared URLs, so packed funders
    are where paraphrased titles become `ambiguous_title` false positives.
    """
    funder_mean_f1 = _funder_mean_f1(data)
    out = [
        "STRUCTURAL COMPLEXITY (D-E) — rows_per_url; EXPLORATORY and partly a "
        "measurement artefact (see funders.py: the match key consults titles only on "
        "shared URLs, so a packed funder is exactly where paraphrase produces the "
        "largest false-positive class, ambiguous_title)"
    ]
    ordered = sorted(profiles.values(), key=lambda p: -p.rows_per_url)
    funder_w = _col_width("funder", (p.display_name for p in ordered))
    out.append(f"  {_ljust('funder', funder_w)}{'rows/url':>9}{'band':>14}{'mean F1 (all runs)':>21}")
    for p in ordered:
        band = fm.complexity_band(p)
        out.append(
            f"  {_ljust(p.display_name, funder_w)}{p.rows_per_url:>9.2f}{band:>14}"
            f"{_fmt(funder_mean_f1.get(p.display_name)):>21}"
        )
    return "\n".join(out)


def frontier_table(points) -> str:
    """One row per config: realised cost, macro F1, frontier membership and the
    resampled probability of being on the frontier.

    Costs only cover billed runs, so configs that failed more look cheaper. The number
    of unbilled runs is shown as a column and noted for frontier configs.
    """
    ordered = sorted(points, key=lambda p: p.cost_usd)
    config_w = _col_width("config", (p.config_label for p in ordered))

    out = [
        "COST-EFFECTIVENESS FRONTIER — realised spend; "
        "P(frontier) is a funder-clustered bootstrap membership probability"
    ]
    out.append(
        f"  {_ljust('config', config_w)}{'cost $':>9}{'macroF1':>9}{'frontier':>9}"
        f"{'P(frontier)':>13}{'cost excl.':>11}"
    )
    for p in ordered:
        out.append(
            f"  {_ljust(p.config_label, config_w)}{p.cost_usd:>9.4f}{p.macro_f1:>9.3f}"
            f"{'yes' if p.on_frontier else 'no':>9}"
            f"{_fmt(p.frontier_probability, 2):>13}"
            f"{p.n_cost_excluded:>11}"
        )

    on_frontier = sorted(
        (p for p in points if p.on_frontier), key=lambda p: p.cost_usd
    )
    out.append("")
    out.append(
        "'cost excl.' is how many of a config's runs died before billing (D-D) and "
        "so contributed nothing to its cost total above: costs are realised totals "
        "over unequal denominators, not directly comparable run-for-run. On the "
        "frontier: "
        + "; ".join(
            f"{p.config_label} excludes {p.n_cost_excluded} run(s) from its total"
            for p in on_frontier
        )
        + "."
    )
    return "\n".join(out)


def reliability_table(data: ds.AnalysisDataset, scored_dir: Path = ds.DEFAULT_SCORED_DIR) -> str:
    """Mean stable-rate and mean Fleiss' kappa per config, from `configs.json`."""
    configs = _load_configs(scored_dir)

    def label_of(c: dict) -> str:
        cfg = c["config"]
        return ds.config_label(cfg["model"], cfg["prompt_strategy"], cfg["temperature"])

    ordered = sorted(configs, key=label_of)
    labels = [label_of(c) for c in ordered]
    config_w = _col_width("config", labels)

    out = ["RELIABILITY (RQ3) — mean stable-rate and mean Fleiss' kappa across funders, per config"]
    out.append(f"  {_ljust('config', config_w)}{'stable-rate':>13}{'kappa':>9}")
    for c, label in zip(ordered, labels):
        out.append(
            f"  {_ljust(label, config_w)}{_fmt(c['mean_stable_rate']):>13}{_fmt(c['mean_fleiss_kappa']):>9}"
        )
    return "\n".join(out)


def failure_summary(data: ds.AnalysisDataset, scored_dir: Path = ds.DEFAULT_SCORED_DIR) -> str:
    """Failure and abstention counts by kind and model.

    Context overflow could be inefficient search or a site too big for the window; both
    readings are stated and neither is chosen.
    """
    b = sens.failure_breakdown(data, scored_dir=scored_dir)
    out = [
        f"{b.n_runs} runs total: {b.n_failed} failed, {b.n_abstained} abstained "
        "(abstention is model behaviour, not a failure, and is not counted among the "
        f"{b.n_failed})."
    ]
    out.append(
        f"  {b.n_context_overflow} context overflows, {b.n_timed_out} timeouts "
        f"({b.n_context_overflow} + {b.n_timed_out} = {b.n_context_overflow + b.n_timed_out}"
        f"{'' if b.n_context_overflow + b.n_timed_out == b.n_failed else ' != n_failed — see below'})."
    )
    for model, n in sorted(b.by_model.items()):
        out.append(f"  {model}: {n} failed run(s)")
    out.append(
        "context_overflow is genuinely ambiguous — inefficient search burning the "
        "context window (model behaviour) versus a site too large for a 128k window "
        "regardless of search quality (infrastructure); a sample of one model per "
        "failure type cannot separate the two readings, and this report does not "
        "resolve it."
    )
    return "\n".join(out)


def sensitivity_table(comparisons) -> str:
    """Macro F1 per config with and without failed runs, and how many were excluded."""
    ordered = sorted(comparisons, key=lambda c: c.label)
    config_w = _col_width("config", (c.label for c in ordered))

    out = [
        "SENSITIVITY (D-D) — macro F1 with vs without failed "
        "(timed-out / context-overflow) runs"
    ]
    out.append(f"  {_ljust('config', config_w)}{'primary':>9}{'survivors':>10}{'delta':>9}{'n excl.':>9}")
    for c in ordered:
        out.append(
            f"  {_ljust(c.label, config_w)}{_fmt(c.primary):>9}{_fmt(c.sensitivity):>10}"
            f"{_fmt(c.delta):>9}{c.n_excluded:>9}"
        )
    return "\n".join(out)


def _short_label(label: str) -> str:
    """Shorten the model part of a model ID or config label to its last segment."""
    head, sep, rest = label.partition(" | ")
    return f"{head.rsplit('/', 1)[-1]}{sep}{rest}"


def fp_reason_table(profiles) -> str:
    """False positives split by the scorer's own reason classification.

    Shares are of each group's own false positives, not the whole grid's.
    """
    ordered = sorted(profiles, key=lambda p: -p.n_false_positives)
    label_w = _col_width("group", (_short_label(p.label) for p in ordered))
    reasons = sorted(
        {r for p in profiles for r in p.by_reason},
        key=lambda r: -sum(p.by_reason.get(r, 0) for p in profiles),
    )

    out = [
        "FALSE-POSITIVE COMPOSITION — counts, and each reason's share of that "
        "group's own false positives"
    ]
    header = f"  {_ljust('group', label_w)}{'n FP':>7}"
    for r in reasons:
        header += f"{r:>20}"
    out.append(header)
    for p in ordered:
        row = f"  {_ljust(_short_label(p.label), label_w)}{p.n_false_positives:>7}"
        for r in reasons:
            n = p.by_reason.get(r)
            cell = DASH if n is None else f"{n} ({p.share[r]:.0%})"
            row += f"{cell:>20}"
        out.append(row)

    out.append("")
    out.append(
        "`ambiguous_title` is charged when a candidate's URL is shared by more than "
        "one ground-truth row and its title does not match — which is exactly the "
        "'packed' funders of D-E (funders.py). Its size here is therefore the measure "
        "of how much of the structural-complexity result is scoring strictness rather "
        "than page difficulty; this table reports the composition and does not resolve "
        "that question."
    )
    return "\n".join(out)


def fabrication_table(profiles) -> str:
    """Fabricated items and mean hallucination rate per group.

    `n rate def.` is shown because runs that extracted nothing have no rate, so the
    denominator varies a lot between groups."""
    ordered = sorted(profiles, key=lambda p: -p.total_fabricated)
    label_w = _col_width("group", (_short_label(p.label) for p in ordered))

    out = [
        "FABRICATION — invented items; a fabricated funding URL is the worst output "
        "class for this product, worse than a miss"
    ]
    out.append(
        f"  {_ljust('group', label_w)}{'n runs':>9}{'n fabricated':>14}"
        f"{'runs fabricating':>18}{'mean halluc.':>14}{'n rate def.':>13}"
    )
    for p in ordered:
        out.append(
            f"  {_ljust(_short_label(p.label), label_w)}{p.n_runs:>9}"
            f"{p.total_fabricated:>14}{p.n_runs_fabricating:>18}"
            f"{_fmt(p.mean_hallucination_rate, 4):>14}{p.n_rate_defined:>13}"
        )
    out.append("")
    out.append(
        "'mean halluc.' averages only over runs with a DEFINED hallucination rate: a "
        "run that extracted nothing has no rate, and coercing that to 0 would read as "
        "scrupulous rather than silent. '" + DASH + "' means no run in the group had "
        "a defined rate."
    )
    return "\n".join(out)


def latency_table(profiles) -> str:
    """Wall-clock latency per group."""
    ordered = sorted(profiles, key=lambda p: (p.p50_s is None, p.p50_s))
    label_w = _col_width("group", (_short_label(p.label) for p in ordered))

    out = ["LATENCY (seconds) — realised wall-clock per run"]
    out.append(
        f"  {_ljust('group', label_w)}{'n runs':>9}{'n defined':>11}{'mean':>10}"
        f"{'p50':>10}{'p95':>11}{'max':>11}"
    )
    for p in ordered:
        out.append(
            f"  {_ljust(_short_label(p.label), label_w)}{p.n_runs:>9}{p.n_defined:>11}"
            f"{_fmt(p.mean_s, 1):>10}{_fmt(p.p50_s, 1):>10}"
            f"{_fmt(p.p95_s, 1):>11}{_fmt(p.max_s, 1):>11}"
        )
    out.append("")
    out.append(
        "A run that died of context overflow has no defined latency (there is no "
        "completion time for a run that never completed) and is excluded from every "
        "statistic here; a timed-out run's latency IS defined but is the cap, not a "
        "measurement of the work, so it should be read as censored. A mean far above "
        "the p50 means the group is usually fast and occasionally catastrophic — the "
        "shape a mean alone would hide. Percentiles use numpy's default linear "
        "interpolation."
    )
    return "\n".join(out)


def abstention_table(profiles) -> str:
    """How often each group declined to answer. Not a failure, but it still gives
    zero recall for that funder."""
    ordered = sorted(profiles, key=lambda p: (p.rate is None, -(p.rate or 0.0)))
    label_w = _col_width("group", (_short_label(p.label) for p in ordered))

    out = [
        "ABSTENTION — runs that correctly reported nothing; never counted as failures "
        "(D-D), but recall is 0 for that funder either way"
    ]
    out.append(
        f"  {_ljust('group', label_w)}{'n runs':>9}{'n abstained':>13}{'rate':>9}"
    )
    for p in ordered:
        out.append(
            f"  {_ljust(_short_label(p.label), label_w)}{p.n_runs:>9}"
            f"{p.n_abstained:>13}{_fmt(p.rate, 3):>9}"
        )
    out.append("")
    out.append(
        "'Found nothing' and 'found the wrong thing' are different product failures "
        "with different fixes; F1 alone pools them. A group with a high abstention "
        "rate has part of its F1 deficit explained by refusal rather than by error."
    )
    return "\n".join(out)


def cost_per_tp_table(rows) -> str:
    """Realised spend per correctly found opportunity.

    Unlike the frontier totals, which depend on 80 runs per config, this carries over
    to other workloads."""
    ordered = sorted(rows, key=lambda r: (r.cost_per_tp is None, r.cost_per_tp))
    label_w = _col_width("config", (r.label for r in ordered))

    out = [
        "COST PER TRUE POSITIVE — realised spend per recovered opportunity; the unit "
        "cost that survives a change of denominator"
    ]
    out.append(
        f"  {_ljust('config', label_w)}{'cost $':>10}{'n TP':>8}{'$ / TP':>10}"
        f"{'cost excl.':>12}"
    )
    for r in ordered:
        out.append(
            f"  {_ljust(r.label, label_w)}{r.cost_usd:>10.4f}{r.n_true_positives:>8}"
            f"{_fmt(r.cost_per_tp, 4):>10}{r.n_cost_excluded:>12}"
        )
    out.append("")
    out.append(
        "A config that recovered nothing has an UNDEFINED unit cost ('" + DASH + "'), "
        "never 0 and never an infinity: inventing a number there would rank a config "
        "that found nothing as the cheapest on the page. 'cost excl.' carries "
        "frontier.py's caveat forward — runs that died before billing contribute "
        "nothing to the cost total, so these totals sit over unequal denominators."
    )
    return "\n".join(out)


def _ci_cell(ci) -> str:
    """A bootstrap interval as one field, or a dash when undefined."""
    if ci is None or ci.undefined or ci.lower is None or ci.upper is None:
        return DASH
    return f"[{ci.lower:+.3f}, {ci.upper:+.3f}]"


def edge_case_table(contrasts, alone: bool = False) -> str:
    """Recall on each edge-case tag against the UNTAGGED baseline, paired within funder.

    'tag clu.' is funders with any tagged row; 'paired clu.' is funders with both
    tagged and untagged rows, which is what the contrast actually rests on.
    """
    ordered = sorted(contrasts, key=lambda c: -c.n_rows)
    tag_w = _col_width("tag", (c.tag for c in ordered))
    rows_header = "rows alone" if alone else "rows"

    out = [
        "EDGE-CASE CONTRAST (D-C, extended) — mean recall on tagged rows MINUS mean "
        "recall on UNTAGGED rows, paired within funder; negative means the tag costs "
        "recall"
    ]
    if alone:
        out.append(
            "  ALONE reading: the tagged side is restricted to rows carrying ONLY "
            "this tag, removing the co-occurrence confound at the cost of a much "
            "smaller denominator."
        )
    out.append(
        f"  {_ljust('tag', tag_w)}{rows_header:>11}{'tag clu.':>10}{'paired clu.':>13}"
        f"{'tagged':>9}{'untagged':>10}{'diff':>9}{'95% CI':>20}{'status':>14}"
    )
    for c in ordered:
        n_rows = c.n_rows_alone if alone else c.n_rows
        status = "inferential" if c.inferential else "descriptive"
        diff = None if c.difference is None else c.difference.point
        out.append(
            f"  {_ljust(c.tag, tag_w)}{n_rows:>11}{c.n_tag_clusters:>10}"
            f"{c.n_paired_clusters:>13}{_fmt(c.mean_tagged):>9}"
            f"{_fmt(c.mean_untagged):>10}{_fmt(diff):>9}"
            f"{_ci_cell(c.difference):>20}{status:>14}"
        )

    out.append("")
    for c in ordered:
        if c.reason:
            out.append(f"  {c.tag}: {c.reason}")
    out.append("")
    out.append(
        "The baseline is rows carrying NO tag at all, not 'rows without this tag': "
        "the tags overlap heavily (see the overlap table), so 'not multi_deadline' "
        "would be substantially made of other edge cases and the contrast would "
        "compare one edge case against a mixture of others. A funder whose in-scope "
        "rows are ALL tagged has no baseline and drops out of the pairing entirely, "
        "which is why 'paired clu.' is well below 'tag clu.' for every tag."
    )
    return "\n".join(out)


def tag_count_table(bands) -> str:
    """Mean recall grouped by how many edge-case tags a row carries.

    The groups cover every in-scope row exactly once, so the row counts sum to the
    corpus."""
    out = [
        "RECALL BY EDGE-CASE COUNT — does stacking edge cases compound? "
        "EXPLORATORY: the bands were defined after seeing the tag distribution"
    ]
    out.append(
        f"  {_ljust('tags on row', 14)}{'rows':>7}{'clusters':>10}{'mean recall':>13}"
        f"{'95% CI':>22}"
    )
    for b in bands:
        out.append(
            f"  {_ljust(str(b.n_tags), 14)}{b.n_rows:>7}{b.n_clusters:>10}"
            f"{_fmt(b.mean_recall):>13}{_ci_cell(b.ci):>22}"
        )
    out.append("")
    out.append(
        "Intervals here are on each band's own mean, not on a difference between "
        "bands, and they overlap: this table shows the shape, it does not test it."
    )
    return "\n".join(out)


def tag_overlap_table(matrix, split: str = "tune") -> str:
    """Rows shared between each pair of edge-case tags; the diagonal is each tag's
    total. Shown because overlap confounds the per-tag contrasts.

    `split` is only used in the heading.
    """
    tags = sorted(matrix)
    tag_w = _col_width("tag", tags)
    out = [
        f"EDGE-CASE TAG OVERLAP — in-scope {split} rows carrying BOTH tags; the diagonal "
        "is the tag's own row count"
    ]
    header = f"  {_ljust('tag', tag_w)}"
    for t in tags:
        header += f"{t[:12]:>14}"
    out.append(header)
    for a in tags:
        row = f"  {_ljust(a, tag_w)}"
        for b in tags:
            row += f"{matrix[a][b]:>14}"
        out.append(row)
    out.append("")
    out.append(
        "Where two tags share most of their rows, their marginal contrasts are "
        "substantially the same rows measured twice and cannot be read as independent "
        "effects."
    )
    return "\n".join(out)


def field_accuracy_table(rows) -> str:
    """Exact-match accuracy per group on each structured field.

    Accuracy only covers opportunities the run found, so `n eval.` is shown beside each
    figure: the denominators differ by up to four times between models."""
    groups = sorted({r.label for r in rows})
    by = {(r.label, r.field): r for r in rows}
    label_w = _col_width("group", (_short_label(g) for g in groups))

    out = [
        "STRUCTURED FIELD ACCURACY — exact match, pooled over opportunities; "
        "CONDITIONAL ON DETECTION (see 'n eval.')"
    ]
    header = f"  {_ljust('group', label_w)}"
    for f in fld.FIELDS:
        header += f"{f:>18}{'n eval.':>10}"
    out.append(header)
    for g in groups:
        row = f"  {_ljust(_short_label(g), label_w)}"
        for f in fld.FIELDS:
            r = by.get((g, f))
            row += f"{_fmt(r.accuracy if r else None):>18}{(r.n_evaluated if r else 0):>10}"
        out.append(row)

    out.append("")
    out.append(
        "A field is only scorable on an opportunity the run actually found, so these "
        "accuracies sit over different populations and are NOT a like-for-like "
        "ranking: the weakest model's figures rest on the easiest opportunities it "
        "reached. END-TO-END below puts detection and extraction on one denominator, "
        "where that selection effect cannot hide. Opportunities whose field the "
        "labeller could not establish (uncertain_fields) are excluded from both "
        "numerator and denominator."
    )
    return "\n".join(out)


def field_direction_table(rows) -> str:
    """Which way each field's errors run: inventing a value, or omitting one."""
    groups = sorted({r.label for r in rows})
    by = {(r.label, r.field): r for r in rows}
    label_w = _col_width("group", (_short_label(g) for g in groups))

    out = [
        "FIELD ERROR DIRECTION — 'asserted' = agent supplied a value the truth does "
        "not have; 'omitted' = agent left blank a value the truth has"
    ]
    header = f"  {_ljust('group', label_w)}"
    for f in fld.FIELDS:
        header += f"{f + ' asrt/omit':>26}"
    out.append(header)
    for g in groups:
        row = f"  {_ljust(_short_label(g), label_w)}"
        for f in fld.FIELDS:
            r = by.get((g, f))
            cell = DASH if r is None else f"{r.asserted_where_blank} / {r.blank_where_value}"
            row += f"{cell:>26}"
        out.append(row)
    out.append("")
    out.append(
        "The two are different product failures and one accuracy figure pools them: "
        "an asserted value is an invention a user may act on, an omission is a gap "
        "they can see. A field whose errors are overwhelmingly omissions is failing "
        "conservatively; one whose errors are assertions is not."
    )
    return "\n".join(out)


def field_by_tag_table(rows) -> str:
    """Field accuracy on edge-case rows against the untagged baseline, paired within
    funder.

    `edge_case_table` asks whether tagged opportunities are harder to find; this asks
    whether they are harder to read correctly once found.
    """
    ordered = sorted(rows, key=lambda r: (r.tag, fld.FIELDS.index(r.field)))
    tag_w = _col_width("tag", (r.tag for r in ordered))
    field_w = _col_width("field", (r.field for r in ordered))

    out = [
        "FIELD ACCURACY BY EDGE-CASE TAG — accuracy on tagged rows MINUS accuracy on "
        "UNTAGGED rows, paired within funder; negative means the tag costs accuracy"
    ]
    out.append(
        f"  {_ljust('tag', tag_w)}{_ljust('field', field_w)}{'rows':>7}"
        f"{'paired clu.':>13}{'tagged':>9}{'untagged':>10}{'diff':>9}{'95% CI':>20}"
    )
    for r in ordered:
        diff = None if r.difference is None else r.difference.point
        out.append(
            f"  {_ljust(r.tag, tag_w)}{_ljust(r.field, field_w)}{r.n_rows:>7}"
            f"{r.n_paired_clusters:>13}{_fmt(r.accuracy_tagged):>9}"
            f"{_fmt(r.accuracy_untagged):>10}{_fmt(diff):>9}{_ci_cell(r.difference):>20}"
        )
    out.append("")
    out.append(
        "Every row here is DESCRIPTIVE by the D-C threshold applied to paired "
        "clusters (edgecases.py): no tag has 8 funders carrying both tagged and "
        "untagged scorable rows. An interval far from zero on a large effect is still "
        "worth reporting as the strongest available signal, but it is not a "
        "pre-registered test and must not be written up as one."
    )
    return "\n".join(out)


def end_to_end_table(rows) -> str:
    """The share of in-scope opportunities both FOUND and correctly READ, per config.

    Both columns share a denominator, so their gap is what field extraction loses on
    top of detection."""
    ordered = sorted(rows, key=lambda r: -(r.end_to_end_rate or 0.0))
    label_w = _col_width("config", (r.label for r in ordered))
    field = ordered[0].field if ordered else "?"
    n_opp = ordered[0].n_opportunities if ordered else 0
    n_exc = ordered[0].n_excluded_uncertain if ordered else 0

    out = [
        f"END-TO-END RESULT (field = {field}) — share of ALL in-scope opportunities "
        f"both found AND correctly extracted; {n_opp} opportunities, "
        f"{n_exc} excluded as uncertain"
    ]
    out.append(
        f"  {_ljust('config', label_w)}{'detected':>10}{'end-to-end':>12}{'gap':>8}"
        f"{'cost $':>9}{'$/usable':>11}{'frontier':>10}"
    )
    for r in ordered:
        gap = (
            None
            if (r.detection_rate is None or r.end_to_end_rate is None)
            else r.end_to_end_rate - r.detection_rate
        )
        out.append(
            f"  {_ljust(r.label, label_w)}{_fmt(r.detection_rate):>10}"
            f"{_fmt(r.end_to_end_rate):>12}{_fmt(gap):>8}{r.cost_usd:>9.3f}"
            f"{_fmt(r.cost_per_usable, 4):>11}{('yes' if r.on_frontier else 'no'):>10}"
        )
    out.append("")
    out.append(
        "An opportunity that was never found counts as a failure here, which is the "
        "point: you cannot extract a correct deadline from a grant you did not find. "
        "'frontier' is the Pareto frontier over (realised cost, end-to-end rate), "
        "computed by the same domain-free `frontier.pareto_frontier` the F1 frontier "
        "uses — a config on the F1 frontier need not be on this one."
    )
    return "\n".join(out)


def runtime_health_table(health) -> str:
    """Schema validity, retries and token totals."""
    out = ["RUNTIME HEALTH — conformance, retries and token totals"]
    out.append(f"  runs                       {health.n_runs}")
    out.append(
        f"  schema-invalid output      {health.n_schema_invalid} "
        f"({health.n_schema_invalid / health.n_runs:.1%} of runs)"
    )
    for model, n in sorted(health.by_model_schema_invalid.items(), key=lambda kv: -kv[1]):
        out.append(f"      {model.rsplit('/', 1)[-1]:<26}{n}")
    out.append(
        f"  runs retried               {health.n_retried} "
        f"(max attempts {health.max_attempts})"
    )
    out.append(f"  step budget exhausted      {health.n_step_budget_exhausted}")
    out.append(f"  submit failed              {health.n_submit_failed}")
    out.append(f"  input tokens               {health.total_input_tokens:,}")
    out.append(f"  output tokens              {health.total_output_tokens:,}")
    out.append(f"  unattributable false pos.  {health.n_fp_unattributable}")
    out.append("")
    out.append(
        "Token totals are what make cost transferable: dollars are a function of this "
        "grid's prices on the day, tokens are not. 'unattributable false positives' "
        "are extractions matching no ground-truth row at all; score.py already fixes "
        "how precision treats them, and that decision is not restated here — the "
        "count is surfaced so the size of the affected population is visible."
    )
    return "\n".join(out)


def field_by_config_table(rows) -> str:
    """Every config's accuracy on every structured field, sorted by `deadlines`
    accuracy (the weakest field)."""
    by = {(r.label, r.field): r for r in rows}
    configs = sorted(
        {r.label for r in rows},
        key=lambda c: -((by[(c, "deadlines")].accuracy) or 0.0),
    )
    label_w = _col_width("config", configs)

    out = [
        "STRUCTURED FIELD ACCURACY BY CONFIG — exact match; ordered by deadline "
        "accuracy; CONDITIONAL ON DETECTION"
    ]
    header = f"  {_ljust('config', label_w)}"
    for f in fld.FIELDS:
        header += f"{f:>17}"
    out.append(header)
    for c in configs:
        row = f"  {_ljust(c, label_w)}"
        for f in fld.FIELDS:
            row += f"{_fmt(by[(c, f)].accuracy):>17}"
        out.append(row)
    out.append("")
    out.append(
        "Denominators differ by config as well as by model — a config that found "
        "little is scored on the little it found — so read this beside the "
        "END-TO-END table, which puts detection and extraction on one denominator."
    )
    return "\n".join(out)


def field_factor_table(rows) -> str:
    """Field accuracy by prompt strategy (or temperature) WITHIN each model.

    Split by model because model has by far the largest effect, so pooling would
    mostly measure which models are in each cell."""
    ordered = sorted(rows, key=lambda r: (r.model, r.level, fld.FIELDS.index(r.field)))
    factor = ordered[0].factor if ordered else "?"
    models = sorted({r.model for r in ordered})
    levels = sorted({r.level for r in ordered})
    by = {(r.model, r.level, r.field): r for r in ordered}
    level_w = _col_width(factor, levels)

    out = [
        f"STRUCTURED FIELD ACCURACY BY {factor.upper()}, WITHIN MODEL — "
        "EXPLORATORY: no field-level contrast is pre-registered"
    ]
    for model in models:
        out.append(f"  {model.rsplit('/', 1)[-1]}")
        header = f"    {_ljust(factor, level_w)}"
        for f in fld.FIELDS:
            header += f"{f:>17}{'n':>8}"
        out.append(header)
        for level in levels:
            row = f"    {_ljust(level, level_w)}"
            for f in fld.FIELDS:
                r = by.get((model, level, f))
                row += f"{_fmt(r.accuracy if r else None):>17}{(r.n_evaluated if r else 0):>8}"
            out.append(row)
        out.append("")
    out.append(
        "The pre-registered contrasts (D-A) found no prompt effect on DETECTION F1. "
        "They say nothing about extraction, which is a different endpoint measured on "
        "a different population, and nothing here is a test — these are descriptive "
        "cells with unequal denominators, offered because the model-level table pools "
        "them away entirely."
    )
    return "\n".join(out)


def generalisation_note(result: gen.GeneralisationResult) -> str:
    """Tune-to-validation gap per config in the lock file, and whether the ranking held.

    The funder sets do not overlap, so funder counts are shown and no paired test or
    interval is given.
    """
    labels = [g.label for g in result.gaps]
    label_w = _col_width("config", labels)

    out = ["GENERALISATION (RQ6) — locked configs, tune versus held-out validation"]
    out.append(
        f"  {_ljust('config', label_w)}{'tuneF1':>9}{'valF1':>9}{'gap':>9}"
        f"{'tuneRank':>10}{'valRank':>9}{'tuneN':>7}{'valN':>6}"
    )
    for g in result.gaps:
        out.append(
            f"  {_ljust(g.label, label_w)}"
            f"{_fmt(g.tune_macro_f1, 4):>9}{_fmt(g.validation_macro_f1, 4):>9}"
            f"{_fmt(g.gap, 4):>9}"
            f"{DASH if g.tune_rank is None else g.tune_rank:>10}"
            f"{DASH if g.validation_rank is None else g.validation_rank:>9}"
            f"{g.tune_n_funders:>7}{g.validation_n_funders:>6}"
        )
    out.append("")
    out.append(
        "gap = validation macro F1 - tune macro F1; negative means the config did worse "
        "out of sample. The two means are over disjoint funder sets, so this is a "
        "difference of two independent macro means, NOT a paired difference: no "
        "Wilcoxon, no sign test and no bootstrap interval is reported here, because "
        "there is no funder present on both sides to pair on."
    )
    if result.ranking_preserved is True:
        out.append(
            "H4 — RANKING PRESERVED: every locked config holds the same rank on the "
            "held-out split as it held on tune. The configuration chosen on tune is "
            "still the better of the two out of sample, which is the claim the "
            "single-shot run was spent to test."
        )
    elif result.ranking_preserved is False:
        out.append(
            "H4 — RANKING NOT PRESERVED: the locked configs change order between the "
            "splits. The tune ranking did not generalise, and any claim that the "
            "chosen config is the better one rests on the tune split alone."
        )
    else:
        out.append(
            "H4 — not assessed: fewer than two locked configs have a defined macro F1 "
            "on both splits, and a set of one has no ordering to preserve."
        )
    if result.unlocked_validation_configs:
        out.append(
            "configs present in the validation artefact but NOT named by the lock "
            "(these had no pre-registered out-of-sample claim and are excluded above): "
            + ", ".join(result.unlocked_validation_configs)
        )
    out.append(
        f"Provenance: lock {result.lock_path} written {result.locked_at}; tune artefact "
        f"{result.tune_scored_dir}; validation artefact {result.validation_scored_dir}; "
        f"one scorer source_hash {result.source_hash[:12]} across both."
    )
    return "\n".join(out)


def _absent_tags(gt, funders: tuple[str, ...]) -> list[str]:
    """Edge-case tags with no in-scope rows on this split, apart from `recently_closed`."""
    counts = {tag.value: 0 for tag in EdgeCaseTag}
    for funder in funders:
        for row in gt.for_funder(funder).in_scope:
            for tag in row.edge_case_tags:
                counts[getattr(tag, "value", tag)] += 1
    return [
        tag
        for tag, n in sorted(counts.items())
        if n == 0 and tag != _DISTRACTOR_ONLY_TAG
    ]


def scope_notes(
    data: ds.AnalysisDataset, scored_dir: Path = ds.DEFAULT_SCORED_DIR
) -> str:
    """What the output does not cover and why, plus where the numbers came from.

    The split and funder counts are read from the runs so a validation output can't be
    labelled as tune.
    """
    gt = load_ground_truth()
    funders = data.funders()
    n_in_scope = sum(len(gt.for_funder(f).in_scope) for f in funders)
    source_hash = data.manifest["scorer"]["source_hash"]
    n_runs = data.manifest.get("counts", {}).get("n_scored", len(data.runs))
    splits = data.splits()
    split = splits[0] if len(splits) == 1 else "+".join(splits)

    out = ["SCOPE"]
    out.append(
        f"This report covers the {len(funders)} {split} funders and their {n_in_scope} "
        "in-scope ground-truth rows only."
    )
    if split == "validation":
        out.append(
            "This is the HELD-OUT split, run single-shot on the configurations locked "
            "in eval/locked-config.json before any validation page was fetched "
            "(preregistration §5). No configuration was chosen, tuned or discarded "
            "after seeing anything in this report, and none may be: the out-of-sample "
            "claim in the GENERALISATION section above is the whole return on that "
            "restraint and iterating here would spend it. The comparison against "
            "Pivot-RP remains OUT OF SCOPE — no Pivot-RP data exists in the repo to "
            "compare against."
        )
        out.append(
            "Model, temperature and every prompt but the two locked ones are absent by "
            "construction, not by accident: the lock fixed them. Sections that need a "
            "second level of a fixed factor are marked "
            f"'{_NOT_ESTIMABLE.strip()}' rather than omitted, so this report stays "
            "section-for-section comparable with the tune report at eval/figures/."
        )
    else:
        out.append(
            "RQ6 (the tune-to-validation gap) and the comparison against Pivot-RP are OUT "
            "OF SCOPE here: the validation run has not happened yet, "
            "and no Pivot-RP data exists to compare against. A silent gap is worse than a "
            "stated one, so this is stated rather than left implicit."
        )
    out.append(
        "Funder type (D-F) and structural complexity (D-E) were both defined after "
        "seeing the tune data and are exploratory, not confirmatory: neither is ever "
        "statistically tested. Structural complexity is additionally partly a "
        "measurement artefact: the match key (identity.match_key) consults a "
        "candidate's title only when its normalised URL is shared by more than one of "
        "a funder's ground-truth rows, so a 'packed' funder is exactly where an "
        "agent's paraphrase of a title is most likely to be scored ambiguous_title, "
        "the grid's largest false-positive class — not necessarily where the page "
        "is genuinely harder to parse."
    )
    for tag in _absent_tags(gt, funders):
        out.append(f"{tag} has no {split} rows and cannot be assessed on this split.")
    out.append(
        f"Every bootstrap interval in this report is clustered on {len(funders)} funders; at that "
        "cluster count the interval is likely optimistic (Davison & Hinkley 1997; "
        "Field & Welsh 2007), which is why the cluster count is printed beside every "
        "interval rather than a bare confidence level."
    )
    out.append(
        f"Provenance: scorer source_hash {source_hash[:12]} over {n_runs} scored runs "
        f"({Path(scored_dir) / 'manifest.json'})."
    )
    return "\n".join(out)


def full_report(
    data: ds.AnalysisDataset,
    gt,
    profiles: dict,
    family,
    points,
    scored_dir: Path = ds.DEFAULT_SCORED_DIR,
    generalisation: gen.GeneralisationResult | None = None,
) -> str:
    """Every section as one text document.

    `scored_dir` must be the directory `data` was loaded from, since several sections
    re-read its raw files; otherwise they can silently show the tune grid's numbers.
    `generalisation` is the tune-versus-validation result, or `None` for the tune
    output, where it is out of scope.
    """
    comparisons = sens.compare_with_and_without_failures(data)
    fixed_factors = single_level_factors(data)
    sections = [
        ("LEADERBOARD", leaderboard(data, scored_dir=scored_dir)),
        ("STRUCTURED FIELD ACCURACY",
         field_accuracy_table(fld.field_accuracy(data, gt, scored_dir=scored_dir))),
        ("FIELD ACCURACY BY CONFIG",
         field_by_config_table(
             fld.field_accuracy(data, gt, key="config", scored_dir=scored_dir))),
        ("FIELD ACCURACY BY PROMPT",
         field_factor_table(
             fld.field_accuracy_by_factor(data, gt, "prompt", scored_dir=scored_dir))),
        ("FIELD ACCURACY BY TEMPERATURE",
         field_factor_table(
             fld.field_accuracy_by_factor(data, gt, "temperature", scored_dir=scored_dir))),
        ("FIELD ERROR DIRECTION",
         field_direction_table(fld.field_directions(data, scored_dir=scored_dir))),
        ("END-TO-END (DEADLINES)",
         end_to_end_table(fld.end_to_end_frontier(
             fld.end_to_end(data, gt, field="deadlines", scored_dir=scored_dir)))),
        ("END-TO-END (ALL FIELDS)",
         end_to_end_table(fld.end_to_end_frontier(
             fld.end_to_end(data, gt, field="all", scored_dir=scored_dir)))),
        ("CONFIRMATORY CONTRASTS", contrast_table(family, fixed_factors=fixed_factors)),
        ("INTERACTION", interaction_note(data)),
        ("TEMPERATURE", temperature_note(data, family)),
        ("STRATA", stratum_table(data, gt)),
        ("EDGE-CASE CONTRAST", edge_case_table(ec.tag_contrast_family(data, gt))),
        ("EDGE-CASE CONTRAST (ALONE)",
         edge_case_table(ec.tag_contrast_family(data, gt, alone=True), alone=True)),
        ("FIELD ACCURACY BY EDGE-CASE TAG",
         field_by_tag_table([
             fld.field_accuracy_by_tag(data, gt, f, t, scored_dir=scored_dir)
             for t in ("multi_deadline", "restricted", "rolling")
             for f in fld.FIELDS
         ])),
        ("EDGE-CASE COUNT", tag_count_table(ec.recall_by_tag_count(data, gt))),
        ("EDGE-CASE OVERLAP",
         tag_overlap_table(ec.tag_cooccurrence(gt, data.funders()), split=_split_name(data))),
        ("FUNDER TYPE", funder_type_table(data, profiles)),
        ("STRUCTURAL COMPLEXITY", complexity_table(data, profiles)),
        ("COST-EFFECTIVENESS", frontier_table(points)),
        ("RELIABILITY", reliability_table(data, scored_dir=scored_dir)),
        ("COST PER TRUE POSITIVE", cost_per_tp_table(tel.cost_per_true_positive(data))),
        ("LATENCY", latency_table(tel.latency_profiles(data))),
        ("ABSTENTION", abstention_table(tel.abstention_profiles(data))),
        ("FALSE-POSITIVE COMPOSITION", fp_reason_table(err.fp_reason_totals(data))),
        ("FABRICATION", fabrication_table(err.fabrication_totals(data))),
        ("SENSITIVITY",
         failure_summary(data, scored_dir=scored_dir) + "\n\n" + sensitivity_table(comparisons)),
        ("RUNTIME HEALTH", runtime_health_table(tel.runtime_health(scored_dir=scored_dir))),
    ]
    if generalisation is not None:
        sections.append(("GENERALISATION", generalisation_note(generalisation)))
    sections.append(("SCOPE", scope_notes(data, scored_dir=scored_dir)))
    out: list[str] = []
    for heading, body in sections:
        out.append("#" * 80)
        out.append(heading)
        out.append("#" * 80)
        out.append(body)
        out.append("")
    return "\n".join(out)
