"""Tests for the grid preflight checks and provider-pin verification.

Each failing-check test has a matching test showing the same check passes on good
input. `assert_provider_pin` is tested against the live `config.MODEL_ROUTING`.

"""
from __future__ import annotations

from pathlib import Path

import pytest

from grant_eval import config
from grant_eval.dataset.registry import load_registry
from grant_eval.grid.cells import enumerate_cells
from grant_eval.grid.preflight import (
    PreflightError,
    ProviderPinMismatch,
    assert_provider_pin,
    enforce,
    preflight,
)
from grant_eval.schema import Split

pytestmark = pytest.mark.usefixtures("git_checkout")


@pytest.fixture(autouse=True)
def _fresh_prompt_fingerprints():
    """Some tests patch `render`, so drop any fingerprint cached while it was patched."""
    from grant_eval.agent.extract import prompt_fingerprint

    prompt_fingerprint.cache_clear()
    yield
    prompt_fingerprint.cache_clear()

REG = load_registry(Path("data/funders.json"))


def _minimal_plan():
    return enumerate_cells(REG, Split.TUNE, funders=["leverhulme"],
                           models=["nova-lite"], prompts=["minimal"], replicates=[1])


def _mock_factory():
    from grant_eval.agent.router import MockBackend
    return lambda: MockBackend(replies=[])


def _named(checks, name):
    return next(c for c in checks if c.name == name)


def test_minimal_only_plan_passes_the_scope_sensitive_checks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    enforce(checks, scoped=True)  # must not raise


def test_unimplemented_strategy_fails_preflight(tmp_path, monkeypatch):
    """A strategy whose `render` raises fails preflight.

    All four strategies render now, so `render` is patched to raise to reach the failing side.

    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    import grant_eval.agent.prompts as prompts

    real = prompts.render

    def unwritten(strategy, **kwargs):
        if strategy == "chain_of_thought":
            raise NotImplementedError(strategy)
        return real(strategy, **kwargs)

    monkeypatch.setattr(prompts, "render", unwritten)
    plan = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"],
                           models=["nova-lite"], prompts=["chain_of_thought"],
                           replicates=[1])
    checks = preflight(plan, registry=REG, scoped=True, results_root=tmp_path,
                       backend_factory=_mock_factory())
    assert _named(checks, "prompt_strategies_render").ok is False
    with pytest.raises(PreflightError, match="prompt_strategies_render"):
        enforce(checks, scoped=True)


def test_implemented_strategy_still_passes_prompt_strategies_render(tmp_path, monkeypatch):
    """`minimal` is implemented, so the check passes for it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "prompt_strategies_render").ok is True


def test_all_four_prompt_strategies_pass_preflight(tmp_path, monkeypatch):
    """A plan spanning all four levels clears `prompt_strategies_render`."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    plan = enumerate_cells(REG, Split.TUNE, funders=["leverhulme"],
                           models=["nova-lite"], replicates=[1])
    assert {cell.prompt_strategy for cell in plan} == set(config.PROMPT_STRATEGIES)
    checks = preflight(plan, registry=REG, scoped=True, results_root=tmp_path,
                       backend_factory=_mock_factory())
    assert _named(checks, "prompt_strategies_render").ok is True


def test_missing_api_key_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "api_key_present").ok is False


def test_present_api_key_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "api_key_present").ok is True


def test_skeleton_backend_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from grant_eval.scoring.skeleton import _BoundedOpenRouterBackend

    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path,
                       backend_factory=lambda: _BoundedOpenRouterBackend(4000))
    assert _named(checks, "no_skeleton_accommodations").ok is False
    with pytest.raises(PreflightError):
        enforce(checks, scoped=True)


def test_mock_backend_is_not_the_skeleton_accommodation(tmp_path, monkeypatch):
    """The check singles out `_BoundedOpenRouterBackend`, not every non-production backend."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "no_skeleton_accommodations").ok is True


def test_ralph_max_iterations_one_is_not_a_skeleton_accommodation(tmp_path, monkeypatch):
    """`max_iterations = 1` is the pre-registered value and is accepted.

    Set explicitly rather than read from the live config, so the test still holds if
    `config.RALPH` changes.

    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setitem(config.RALPH, "max_iterations", 1)
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "no_skeleton_accommodations").ok is True


def test_the_skeleton_check_does_not_import_skeleton_to_do_it(tmp_path, monkeypatch):
    """The check does not import `scoring.skeleton`.

    That module subclasses `OpenRouterBackend` at import time, so importing it while a
    test has patched `OpenRouterBackend` raises a metaclass TypeError. Removing it from
    `sys.modules` reproduces the problem.

    """
    import sys

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delitem(sys.modules, "grant_eval.scoring.skeleton", raising=False)
    monkeypatch.setattr(
        "grant_eval.agent.router.OpenRouterBackend", lambda: object()
    )

    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "no_skeleton_accommodations").ok is True


def test_settled_ralph_policies_are_only_what_run_ralph_implements():
    """The settled set is exactly {"final"}, the only policy run_ralph implements.

    Any other label would produce results named one thing but scored as `final`.

    """
    from grant_eval.grid.preflight import _SETTLED_RALPH_POLICIES

    assert _SETTLED_RALPH_POLICIES == frozenset({"final"})


def test_an_unimplemented_ralph_policy_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setitem(config.RALPH, "scored_iteration_policy", "best_of")
    checks = preflight(_minimal_plan(), registry=REG, scoped=False,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "ralph_policy_settled").ok is False
    with pytest.raises(PreflightError, match="ralph_policy_settled"):
        enforce(checks, scoped=False)


def test_scoped_run_only_warns_on_non_scope_sensitive_failures(tmp_path, monkeypatch):
    """This lets the dispatcher be smoke-tested while gates are still open."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setitem(config.RALPH, "scored_iteration_policy", "_TBD_")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "ralph_policy_settled").ok is False
    enforce(checks, scoped=True)          # warns, does not raise
    with pytest.raises(PreflightError, match="ralph_policy_settled"):
        enforce(checks, scoped=False)     # the real grid refuses


def test_settled_ralph_policy_passes(tmp_path, monkeypatch):
    """The check passes for `final`, the policy run_ralph implements."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setitem(config.RALPH, "scored_iteration_policy", "final")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "ralph_policy_settled").ok is True


def test_content_hash_mismatch_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    tampered = dict(REG)
    tampered["leverhulme"] = REG["leverhulme"].model_copy(
        update={"content_hash": "0" * 64}
    )
    checks = preflight(_minimal_plan(), registry=tampered, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "content_hash_matches").ok is False
    with pytest.raises(PreflightError, match="content_hash_matches"):
        enforce(checks, scoped=True)


def test_matching_content_hash_passes(tmp_path, monkeypatch):
    """The untouched registry passes the check."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "content_hash_matches").ok is True


def test_entry_url_in_mirror_passes_for_the_real_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "entry_url_in_mirror").ok is True


def test_entry_url_not_in_mirror_is_detected(tmp_path, monkeypatch):
    """An entry URL the mirror never captured fails, catching homepage vs crawl-entry mix-ups."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    tampered = dict(REG)
    tampered["leverhulme"] = REG["leverhulme"].model_copy(
        update={"entry_url": "https://www.leverhulme.ac.uk/this-page-was-never-crawled"}
    )
    checks = preflight(_minimal_plan(), registry=tampered, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "entry_url_in_mirror").ok is False
    with pytest.raises(PreflightError, match="entry_url_in_mirror"):
        enforce(checks, scoped=True)


def test_the_validation_plan_passes_preflight(tmp_path, monkeypatch):
    """Every entry in the validation plan passes `entry_url_in_mirror`.

    The manifest's `url` field can be made up from the filename when a lookup misses,
    so the check uses `serve.resolve` instead, which is what decides what the agent is
    served. Checked over the whole plan because the tune plan passes either way.

    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    plan = enumerate_cells(REG, Split.VALIDATION, models=["nova-lite"],
                           prompts=["minimal"], replicates=[1])
    assert {c.funder for c in plan} >= {"fulbright", "heritage-fund"}
    checks = preflight(plan, registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    check = _named(checks, "entry_url_in_mirror")
    assert check.ok is True, check.detail


def test_entry_url_check_uses_the_resolver_the_agent_is_actually_served_through():
    """Every entry_url resolves through `serve.resolve` over `serve.build_lookup`.

    This is the same pair `executor.serve_cell` uses, so the check cannot drift from
    what the agent is actually served.

    """
    import json
    from urllib.parse import urlparse

    from grant_eval.dataset.serve import build_lookup, resolve

    from grant_eval.grid import preflight as pf

    for slug in ("fulbright", "heritage-fund", "leverhulme"):
        record = REG[slug]
        manifest = json.loads(
            (record.snapshot_dir() / "manifest.json").read_text(encoding="utf-8")
        )
        parsed = urlparse(record.entry_url)
        assert resolve(
            build_lookup(record.snapshot_dir(), manifest), parsed.path or "/",
            parsed.query,
        ) is not None
        check = pf._check_entry_url_in_mirror(
            [c for c in enumerate_cells(REG, record.split, funders=[slug],
                                        models=["nova-lite"], prompts=["minimal"],
                                        replicates=[1])],
            REG,
        )
        assert check.ok is True, f"{slug}: {check.detail}"


def test_snapshots_present_passes_for_the_real_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "snapshots_present").ok is True


def test_missing_snapshot_dir_is_detected(tmp_path, monkeypatch):
    """A snapshot date with nothing on disk fails `snapshots_present`."""
    from datetime import date

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    tampered = dict(REG)
    tampered["leverhulme"] = REG["leverhulme"].model_copy(
        update={"snapshot_date": date(1999, 1, 1)}
    )
    checks = preflight(_minimal_plan(), registry=tampered, scoped=True,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "snapshots_present").ok is False
    with pytest.raises(PreflightError, match="snapshots_present"):
        enforce(checks, scoped=True)


def test_full_grid_hard_refuses_on_a_non_scope_sensitive_failure(tmp_path, monkeypatch):
    """An unscoped full-grid run enforces every check, not just the scope-sensitive ones."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setitem(config.RALPH, "scored_iteration_policy", "_TBD_")
    checks = preflight(_minimal_plan(), registry=REG, scoped=False,
                       results_root=tmp_path, backend_factory=_mock_factory())
    with pytest.raises(PreflightError, match="ralph_policy_settled"):
        enforce(checks, scoped=False)


def test_all_checks_pass_for_a_clean_full_grid_invocation(tmp_path, monkeypatch):
    # working_tree_clean depends on the checkout this suite runs in. sleep_inhibited is
    # left out of the OK-check but kept for enforce(), since advisory checks must never raise.
    environment_dependent = {"working_tree_clean", "sleep_inhibited"}
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    checks = preflight(_minimal_plan(), registry=REG, scoped=False,
                       results_root=tmp_path, backend_factory=_mock_factory())
    failed = [c.name for c in checks if not c.ok and c.name not in environment_dependent]
    assert failed == []
    enforce([c for c in checks if c.name != "working_tree_clean"], scoped=False)


# sleep_inhibited is advisory: it warns but never blocks, in either scope.
# `scope_sensitive` can't express that because enforce(scoped=False) treats every
# failure as fatal.
def test_sleep_inhibited_is_advisory():
    import grant_eval.grid.preflight as preflight_mod

    assert preflight_mod._check_sleep_inhibited().advisory is True
    # Other checks must not be advisory, or one could quietly exempt itself.
    non_advisory_checks = [
        preflight_mod._check_ralph_policy_settled(),
        preflight_mod._check_working_tree_clean(),
    ]
    assert all(c.advisory is False for c in non_advisory_checks)


def test_sleep_uninhibited_warns_but_never_blocks_either_scope(tmp_path, monkeypatch, capsys):
    # working_tree_clean is left out because it depends on the local checkout and is
    # not what this test is about.
    import grant_eval.grid.preflight as preflight_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(preflight_mod, "_sleep_is_inhibited", lambda: False)
    checks = [
        c for c in preflight(_minimal_plan(), registry=REG, scoped=True,
                             results_root=tmp_path, backend_factory=_mock_factory())
        if c.name != "working_tree_clean"
    ]
    sleep_check = _named(checks, "sleep_inhibited")
    assert sleep_check.ok is False

    enforce(checks, scoped=True)   # must not raise
    captured = capsys.readouterr()
    assert "sleep_inhibited" in captured.out

    # Even an unscoped full-grid run, which refuses on any other failed check, only
    # warns here.
    enforce(checks, scoped=False)  # must not raise
    captured = capsys.readouterr()
    assert "sleep_inhibited" in captured.out


def test_sleep_inhibited_being_advisory_does_not_shield_other_failures(tmp_path, monkeypatch):
    """An advisory check doesn't hide other failures: a missing API key still blocks a full-grid run."""
    import grant_eval.grid.preflight as preflight_mod

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(preflight_mod, "_sleep_is_inhibited", lambda: False)
    checks = preflight(_minimal_plan(), registry=REG, scoped=False,
                       results_root=tmp_path, backend_factory=_mock_factory())
    with pytest.raises(PreflightError, match="api_key_present") as excinfo:
        enforce(checks, scoped=False)
    assert "sleep_inhibited" not in str(excinfo.value)


def test_sleep_inhibited_true_prints_nothing(tmp_path, monkeypatch, capsys):
    """When the sleep check passes, enforce() prints no warning for it."""
    import grant_eval.grid.preflight as preflight_mod

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(preflight_mod, "_sleep_is_inhibited", lambda: True)
    checks = preflight(_minimal_plan(), registry=REG, scoped=False,
                       results_root=tmp_path, backend_factory=_mock_factory())
    assert _named(checks, "sleep_inhibited").ok is True
    enforce([c for c in checks if c.name != "working_tree_clean"], scoped=False)
    captured = capsys.readouterr()
    assert "sleep_inhibited" not in captured.out


def test_deepseek_flash_is_pinned_to_parasail():
    """deepseek-flash is pinned to Parasail at fp8 with no fallbacks.

    DeepInfra kept returning 429 for this model even at concurrency 1. Novita and
    SiliconFlow are the pre-registered fallbacks.

    """
    provider = config.MODEL_ROUTING["deepseek-flash"]["provider"]
    assert provider["only"] == ["parasail"]
    assert provider["quantizations"] == ["fp8"]
    assert provider["allow_fallbacks"] is False


def test_provider_pin_accepts_the_pinned_provider():
    assert assert_provider_pin("deepseek-flash", "Parasail") is None


def test_provider_pin_rejects_a_substitute():
    with pytest.raises(ProviderPinMismatch, match="deepseek-flash"):
        assert_provider_pin("deepseek-flash", "Together")


def test_provider_pin_ignores_unpinned_arms():
    assert assert_provider_pin("nova-lite", "Amazon Bedrock") is None


def test_provider_pin_tolerates_a_missing_provider_field():
    """Not every response reports a provider; a missing one is not a mismatch."""
    assert assert_provider_pin("deepseek-flash", None) is None


def test_provider_pin_passes_for_a_label_with_no_routing_entry_at_all():
    """An arm with no `MODEL_ROUTING` entry passes whatever provider served it."""
    assert "definitely-not-a-grid-arm" not in config.MODEL_ROUTING
    assert assert_provider_pin("definitely-not-a-grid-arm", "anything at all") is None


def test_provider_pin_normalization_actually_normalizes():
    """Provider names are compared after normalising case and spacing."""
    assert assert_provider_pin("gpt-oss-120b", "Deep Infra") is None


def test_provider_pin_normalization_still_rejects_a_real_difference():
    """A different provider for the same arm still fails."""
    with pytest.raises(ProviderPinMismatch, match="gpt-oss-120b"):
        assert_provider_pin("gpt-oss-120b", "Fireworks")


def test_provider_pin_reads_live_config_not_a_memorised_constant(monkeypatch):
    """The pin is read at call time, not copied at import time.

    The test changes the pin and then passes the value that would have matched the old one.

    """
    assert config.MODEL_ROUTING["nova-lite"]["provider"]["only"] == ["amazon-bedrock"]
    monkeypatch.setitem(
        config.MODEL_ROUTING,
        "nova-lite",
        {"provider": {"only": ["some-other-provider"]}},
    )
    with pytest.raises(ProviderPinMismatch):
        assert_provider_pin("nova-lite", "Amazon Bedrock")


def test_provider_pin_mismatch_carries_expected_and_actual():
    with pytest.raises(ProviderPinMismatch) as excinfo:
        assert_provider_pin("deepseek-flash", "Together")
    assert excinfo.value.actual == "Together"
    assert "parasail" in excinfo.value.expected.lower()


def test_provider_pin_mismatch_is_the_policy_classes_own_class():
    """`preflight.py` re-exports `ProviderPinMismatch` from `grid.policy`, so `classify()` still halts on it."""
    from grant_eval.grid.policy import ProviderPinMismatch as PolicyProviderPinMismatch

    assert ProviderPinMismatch is PolicyProviderPinMismatch
