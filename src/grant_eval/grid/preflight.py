"""Preflight checks run before any cell is dispatched, plus provider-pin checking.

These catch problems that would otherwise show up as quietly wrong results rather
than failures: a prompt that does not render, a snapshot that has drifted, a missing
API key and so on. A full-grid run refuses on any failed check. A scoped run
(`--funder`/`--model`/`--prompt`/`--limit`) only refuses on `scope_sensitive` checks
and warns on the rest. `advisory` checks only ever warn.

| check                     | scope_sensitive | advisory | asserts |
|---------------------------|-----------------|----------|---------|
| prompt_strategies_render  | yes | no | every prompt strategy in `plan` renders |
| snapshots_present         | yes | no | every funder has a snapshot dir + `manifest.json` |
| content_hash_matches      | yes | no | each manifest `content_hash` equals the registry's |
| entry_url_in_mirror       | yes | no | each entry URL resolves via `serve.build_lookup`/`resolve` |
| api_key_present           | yes | no | `OPENROUTER_API_KEY` is set |
| no_skeleton_accommodations| yes | no | `backend_factory()` is not the skeleton's bounded backend |
| ralph_policy_settled      | no  | no | `RALPH["scored_iteration_policy"]` is one `run_ralph` implements |
| working_tree_clean        | no  | no | `freeze.working_tree_clean()` |
| disk_space                | no  | no | free bytes on `results_root` exceed `len(plan) * 1_000_000` |
| sleep_inhibited           | no  | yes | `systemd-inhibit` is in this process's ancestry |

`MODEL_ROUTING` pins the open-weight arms to one provider with fallbacks off.
`assert_provider_pin` is called on every request in `OpenRouterBackend.complete()`
and raises `ProviderPinMismatch` (a HALT) if another provider served it.
Quantisation is pinned in the request but cannot be checked, since OpenRouter does
not report it in the response.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from grant_eval import config
from grant_eval.dataset.registry import FunderRecord
from grant_eval.grid import freeze
from grant_eval.grid.cells import Cell

# Defined in `grid.policy` (it is in the HALT list) and re-exported here.
from grant_eval.grid.policy import ProviderPinMismatch

__all__ = [
    "PreflightError",
    "ProviderPinMismatch",
    "Check",
    "preflight",
    "enforce",
    "assert_provider_pin",
]


class PreflightError(RuntimeError):
    """Raised by `enforce()` when enforced checks fail; the message names each one."""


@dataclass(frozen=True)
class Check:
    """One preflight check's outcome.

    `scope_sensitive` checks are enforced even on scoped runs. `advisory` checks never
    block, whatever the scope.
    """

    name: str
    ok: bool
    detail: str
    scope_sensitive: bool
    advisory: bool = False


_SEPARATORS_RE = re.compile(r"[\s_-]+")


def _normalize_provider_name(name: str) -> str:
    """Lowercase and strip separators, so "Amazon Bedrock" matches "amazon-bedrock"."""
    return _SEPARATORS_RE.sub("", name.strip().lower())


def assert_provider_pin(model_label: str, served_provider: str | None) -> None:
    """Raise `ProviderPinMismatch` if `served_provider` differs from the pinned provider.

    Passes if the response did not report a provider or the model has no pin.
    """
    if served_provider is None:
        return
    routing = config.MODEL_ROUTING.get(model_label, {})
    pinned = routing.get("provider", {}).get("only")
    if not pinned:
        return
    served_norm = _normalize_provider_name(served_provider)
    if any(_normalize_provider_name(candidate) == served_norm for candidate in pinned):
        return
    raise ProviderPinMismatch(
        expected=f"{model_label} pinned to {sorted(pinned)!r}",
        actual=served_provider,
    )


def _check_prompt_strategies_render(plan: list[Cell]) -> Check:
    """Every prompt strategy in `plan` must render without `NotImplementedError`.

    Calls `prompts.render` directly: `prompt_fingerprint` catches that error, so this
    check could never fail if it used that instead.
    """
    from grant_eval.agent import prompts as agent_prompts
    from grant_eval.agent.extract import SENTINEL_PROMPT_CONTEXT

    strategies = sorted({cell.prompt_strategy for cell in plan})
    unimplemented = []
    for strategy in strategies:
        try:
            agent_prompts.render(strategy, **SENTINEL_PROMPT_CONTEXT)
        except NotImplementedError:
            unimplemented.append(strategy)
    ok = not unimplemented
    detail = (
        "all prompt strategies in scope render"
        if ok
        else f"not implemented: {', '.join(unimplemented)}"
    )
    return Check("prompt_strategies_render", ok, detail, scope_sensitive=True)


def _check_snapshots_present(
    plan: list[Cell], registry: dict[str, FunderRecord]
) -> Check:
    """Every funder in `plan` must have a registry entry, snapshot dir and `manifest.json`
    on disk."""
    funders = sorted({cell.funder for cell in plan})
    missing = []
    for slug in funders:
        record = registry.get(slug)
        if record is None:
            missing.append(f"{slug} (no registry entry)")
            continue
        snapshot_dir = record.snapshot_dir()
        manifest_path = snapshot_dir / "manifest.json"
        if not snapshot_dir.is_dir():
            missing.append(f"{slug} (no snapshot dir at {snapshot_dir})")
        elif not manifest_path.is_file():
            missing.append(f"{slug} (no manifest.json at {manifest_path})")
    ok = not missing
    detail = "all snapshots present" if ok else "; ".join(missing)
    return Check("snapshots_present", ok, detail, scope_sensitive=True)


def _load_manifest(record: FunderRecord) -> dict | None:
    manifest_path = record.snapshot_dir() / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _check_content_hash_matches(
    plan: list[Cell], registry: dict[str, FunderRecord]
) -> Check:
    """Each manifest `content_hash` must equal the registry's, catching a mirror changed
    after the registry was written. Funders already missing a snapshot are skipped."""
    funders = sorted({cell.funder for cell in plan})
    mismatched = []
    for slug in funders:
        record = registry.get(slug)
        if record is None:
            continue
        manifest = _load_manifest(record)
        if manifest is None:
            continue
        if manifest.get("content_hash") != record.content_hash:
            mismatched.append(
                f"{slug} (registry={record.content_hash!r}, "
                f"manifest={manifest.get('content_hash')!r})"
            )
    ok = not mismatched
    detail = "all content hashes match" if ok else "; ".join(mismatched)
    return Check("content_hash_matches", ok, detail, scope_sensitive=True)


def _check_entry_url_in_mirror(
    plan: list[Cell], registry: dict[str, FunderRecord]
) -> Check:
    """Each funder's `entry_url` must resolve to a captured file.

    Uses the same `build_lookup` and `resolve` the agent is served through. Comparing
    against `manifest["files"][*]["url"]` is not enough, because some of those URLs are
    rebuilt from filenames and wrongly rejected `fulbright` and `heritage-fund`.
    """
    from grant_eval.dataset.serve import build_lookup, resolve

    funders = sorted({cell.funder for cell in plan})
    absent = []
    for slug in funders:
        record = registry.get(slug)
        if record is None:
            continue
        manifest = _load_manifest(record)
        if manifest is None:
            continue
        parsed = urlparse(record.entry_url)
        lookup = build_lookup(record.snapshot_dir(), manifest)
        if resolve(lookup, parsed.path or "/", parsed.query) is None:
            absent.append(f"{slug} ({record.entry_url!r} does not resolve in the mirror)")
    ok = not absent
    detail = "all entry URLs resolve in their mirror" if ok else "; ".join(absent)
    return Check("entry_url_in_mirror", ok, detail, scope_sensitive=True)


def _check_api_key_present() -> Check:
    ok = bool(os.environ.get("OPENROUTER_API_KEY"))
    detail = "OPENROUTER_API_KEY is set" if ok else "OPENROUTER_API_KEY is not set"
    return Check("api_key_present", ok, detail, scope_sensitive=True)


# The skeleton-only backend, matched by module and class name. Importing
# `scoring.skeleton` here would break offline tests, since it subclasses
# `OpenRouterBackend` at import time and the tests patch that to a function.
_SKELETON_BACKEND: tuple[str, str] = (
    "grant_eval.scoring.skeleton",
    "_BoundedOpenRouterBackend",
)


def _is_skeleton_backend(backend: object) -> bool:
    """Whether `backend` is, or subclasses, the skeleton's bounded backend (by name)."""
    return any(
        (getattr(klass, "__module__", None), klass.__name__) == _SKELETON_BACKEND
        for klass in type(backend).__mro__
    )


def _check_no_skeleton_accommodations(backend_factory: Callable[[], object]) -> Check:
    """Fail if the skeleton's `_BoundedOpenRouterBackend` is in use: it caps `max_tokens`
    and would truncate grid completions."""
    backend = backend_factory()
    is_bounded = _is_skeleton_backend(backend)

    ok = not is_bounded
    detail = (
        "production backend"
        if ok
        else "backend_factory() returns _BoundedOpenRouterBackend (skeleton-only)"
    )
    return Check("no_skeleton_accommodations", ok, detail, scope_sensitive=True)


# Only policies `run_ralph` actually implements. It always scores the final iteration;
# `best_of` would need ground truth to pick, so it was dropped.
_SETTLED_RALPH_POLICIES = frozenset({"final"})


def _check_ralph_policy_settled() -> Check:
    policy_value = config.RALPH.get("scored_iteration_policy")
    ok = policy_value in _SETTLED_RALPH_POLICIES
    detail = (
        f"scored_iteration_policy={policy_value!r}"
        if ok
        else f"scored_iteration_policy={policy_value!r} is not one of "
        f"{sorted(_SETTLED_RALPH_POLICIES)} — the only policies run_ralph implements"
    )
    return Check("ralph_policy_settled", ok, detail, scope_sensitive=False)


def _check_working_tree_clean() -> Check:
    ok = freeze.working_tree_clean()
    detail = "working tree clean" if ok else "tracked files have uncommitted changes"
    return Check("working_tree_clean", ok, detail, scope_sensitive=False)


def _check_disk_space(plan: list[Cell], results_root: Path) -> Check:
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(results_root).free
    needed_bytes = len(plan) * 1_000_000
    ok = free_bytes > needed_bytes
    detail = f"{free_bytes} bytes free at {results_root}, need > {needed_bytes}"
    return Check("disk_space", ok, detail, scope_sensitive=False)


def _sleep_is_inhibited() -> bool:
    """Best effort check for `systemd-inhibit` among this process's ancestors.

    Reads `/proc` directly, and assumes inhibited if `/proc` is missing, since the
    check is only advisory.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return True
    pid = os.getpid()
    seen: set[int] = set()
    while pid and pid not in seen:
        seen.add(pid)
        try:
            comm = (proc / str(pid) / "comm").read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if "systemd-inhibit" in comm:
            return True
        try:
            stat = (proc / str(pid) / "stat").read_text(encoding="utf-8")
            ppid = int(stat.rsplit(")", 1)[-1].split()[1])
        except (OSError, IndexError, ValueError):
            return False
        if ppid == pid:
            return False
        pid = ppid
    return False


def _check_sleep_inhibited() -> Check:
    ok = _sleep_is_inhibited()
    detail = (
        "systemd-inhibit found in process ancestry"
        if ok
        else "systemd-inhibit not detected; run under "
        "'systemd-inhibit --what=sleep:idle' for an unattended multi-night grid"
    )
    # Only a warning: a run that sleeps can be resumed, and there are valid setups this
    # cannot detect.
    return Check("sleep_inhibited", ok, detail, scope_sensitive=False, advisory=True)


def preflight(
    plan: list[Cell],
    *,
    registry: dict[str, FunderRecord],
    scoped: bool,
    results_root,
    backend_factory: Callable[[], object],
) -> list[Check]:
    """Run every check against `plan` and return all the results.

    `scoped` is accepted for symmetry with `enforce()` but not used: every check always
    runs, and only `enforce()` decides what a failure means.
    """
    del scoped
    return [
        _check_prompt_strategies_render(plan),
        _check_snapshots_present(plan, registry),
        _check_content_hash_matches(plan, registry),
        _check_entry_url_in_mirror(plan, registry),
        _check_api_key_present(),
        _check_no_skeleton_accommodations(backend_factory),
        _check_ralph_policy_settled(),
        _check_working_tree_clean(),
        _check_disk_space(plan, results_root),
        _check_sleep_inhibited(),
    ]


def enforce(checks: list[Check], *, scoped: bool) -> None:
    """Raise `PreflightError` for failed checks that matter at this scope; warn on the rest.

    Advisory failures only warn. Otherwise a full run fails on any check, a scoped run
    only on `scope_sensitive` ones.
    """
    failed = [check for check in checks if not check.ok]
    if not failed:
        return

    advisory_failures = [check for check in failed if check.advisory]
    for check in advisory_failures:
        print(f"preflight warning: {check.name} — {check.detail}")

    blocking_failures = [check for check in failed if not check.advisory]
    if not blocking_failures:
        return

    if not scoped:
        names = ", ".join(check.name for check in blocking_failures)
        raise PreflightError(f"preflight failed (full grid): {names}")

    fatal = [check for check in blocking_failures if check.scope_sensitive]
    warnings = [check for check in blocking_failures if not check.scope_sensitive]
    for check in warnings:
        print(f"preflight warning: {check.name} — {check.detail}")
    if fatal:
        names = ", ".join(check.name for check in fatal)
        raise PreflightError(f"preflight failed (scoped): {names}")
