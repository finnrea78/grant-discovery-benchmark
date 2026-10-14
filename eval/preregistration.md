# Pre-registration: grant-discovery evaluation

This is the analysis plan for the evaluation, written and fixed on 2026-09-04 before the
final grid runs, with one later amendment dated 2026-09-13 (section 6a). Analysis
choices made after seeing data are reported as exploratory.

## 1. Primary endpoint
Opportunity-detection **F1** at the (funder, opportunity) level. All confirmatory tests
target F1. Other measures (precision, recall, field accuracy, hallucination rate,
$/correct, latency, consistency) are secondary and reported with intervals.

**The match key.** An extracted record matches a ground-truth row on
`(funder, normalised_url, normalised_title)`, with **the title consulted only to
disambiguate when a normalised URL is shared by more than one row for that funder**
(settled 2026-08-28; also Rule 1 of the labelling protocol). URL alone cannot be the
key: **15 in-scope rows share a URL with another row of the same funder** (8
`society-antiquaries`, 5 `henry-moore`, 2 `prs-foundation`), and the nova-lite skeleton
cell produced the same collision from the extraction side.

**Ambiguity is computed over all of a funder's rows, in scope and distractor alike.**
Counting distractors, **26 rows** share a URL with another row of the same funder, and
three of those shared-URL groups mix in-scope rows with `in_scope = no` distractors
(`henry-moore`, `prs-foundation`, `society-antiquaries`). `wolfson` has six
out-of-scope rows on a single page. If the test "is this URL ambiguous for this
funder?" were run over the in-scope rows alone, a distractor would get the same
URL-only key as a real row, so the test has to see the whole file. Checked against the
CSVs and pinned by a test. (An earlier figure of 13 in-scope rows on 2 URLs was out of
date rather than wrong: the `prs-foundation` pair, `ppl-momentum-accelerator-liverpool`
and `ppl-momentum-accelerator-tees-valley`, both on the shared
`.../ppl-momentum-accelerator-guidance` page, arrived with batch 3 after the original
count was taken.)
The `url` column stays one canonical **live** URL per row, with no synthetic
`#fragment`, so the link back to `source_page` is kept. `agent/ralph.py:identity_key`
uses the same key, so the Ralph loop and the scorer agree on what counts as the same
opportunity.

## 2. Grid (independent variables)
- **Model (3 levels).** Revised 2026-08-23; the commercial-reference arm was swapped on
  2026-08-28. Checked against the live OpenRouter catalogue on 2026-08-23; **to be
  re-checked at freeze, with the date recorded.** Provider and quantization are part
  of the pin. OpenRouter spreads requests for these open-weight models across 20-30
  providers at fp4/fp8/bf16, which would otherwise put an uncontrolled confound on the
  primary independent variable.

  | Arm | Pinned slug | Provider | Quantization | `reasoning_effort` |
  |-----|-------------|----------|--------------|--------------------|
  | deepseek-flash | `deepseek/deepseek-v4-flash-0731` | parasail (amended 2026-09-03, was deepinfra) | fp8 | medium |
  | gpt-oss-120b | `openai/gpt-oss-120b` | deepinfra | bf16 | medium |
  | nova-lite | `amazon/nova-lite-v1` | amazon-bedrock | n/a | none (the arm has no reasoning mode) |

  nova-lite is the commercial reference: closed weights hosted on Bedrock by a single
  provider, so there is no quantization to pin, and its `MODEL_ROUTING` entry has no
  `reasoning_effort` because the model has no reasoning mode.

  **Amendment 2026-09-03: deepseek-flash re-pinned from `deepinfra` to `parasail`.**
  DeepInfra kept returning HTTP 429 for this model, even at concurrency 1:
  `provider_error_code "engine_overloaded"`, `limit_source
  "upstream_provider_shared_pool"`. Direct probes ruled out the account (paid tier, no
  key or account rate limit), the `quantizations` filter (same failure without it) and
  DeepInfra in general (`gpt-oss-120b` on DeepInfra at bf16 returns 200). The cause is
  OpenRouter's shared capacity pool for this one model on this one provider. Probed the
  same day with the pin applied, `quantizations: ["fp8"]`, `allow_fallbacks: false`:

  | Provider | Single request | 8 concurrent | Mean latency (3 runs, 200 max_tokens) | Context |
  |---|---|---|---|---|
  | deepinfra | 429 | n/a | n/a | 1,048,576 |
  | parasail | 200 | 8/8 | 3.30 s | 1,048,576 |
  | novita | 200 | 8/8 | 3.24 s | 1,048,576 |
  | siliconflow | 200 | 8/8 | 3.31 s | 1,048,576 |
  | coreweave | 200 | 8/8 | 3.23 s | 262,144 |
  | baseten | 429 | n/a | n/a | 1,048,576 |

  Parasail, Novita and SiliconFlow could not be told apart on anything measured, so
  **Parasail was picked arbitrarily from equals**; no claim is made that it is the best.
  **Novita, then SiliconFlow, are the pre-registered fallbacks**, so if this happens
  again no new decision is needed. The pre-registered quantization (`fp8`) and
  `allow_fallbacks: false` are unchanged, so the control (pin provider and
  quantization, no fallbacks) still holds; only the pinned provider moved.
  `config_hash` is unaffected (`MODEL_ROUTING` is not in its payload), so no output path
  moves. `result_sha` does move, which is correct: `config.py` affects results and no
  scored grid run predates the change.

  **`gpt-oss-120b` fallback pin: AkashML.** Only two endpoints serve this model at
  `bf16`, AkashML and DeepInfra; every other endpoint is fp4, fp8 or unknown. It is the
  narrowest pin in the design, with exactly one alternative, so AkashML is
  pre-registered as its fallback rather than chosen mid-grid.

  `allow_fallbacks: false` on every arm. These values are set in `config.MODEL_ROUTING`
  and used by the router (commit 6c587f7). **Still to do before freeze:** confirm the pin
  actually applies by reading the served provider and quantization off a live
  response, since the request-side setting alone is not evidence. **This check must be a
  live completion per arm, not a lookup in the endpoints catalogue.** The catalogue does
  not report capacity: on 2026-09-03 it listed DeepInfra for `deepseek-v4-flash-0731`
  with `status = 0` (healthy) at the same time that endpoint was returning 429 on every
  request. A catalogue check would have passed an arm that could not serve a single
  cell.

  Rejected: `openai/gpt-5.6-luna` (accepts no `temperature` parameter, and temperature
  is a pre-registered factor, so it cannot be crossed with it); Gemini 2.0 Flash
  (delisted from OpenRouter); `anthropic/claude-haiku-4.5` (held the commercial-reference
  slot from 2026-08-23 and was dropped on 2026-08-28 **on cost, not capability**:
  $1.00/$5.00 against nova-lite's $0.06/$0.24 per MTok, roughly 17x the input and 21x
  the output price, which is what made the grid look unaffordable);
  `google/gemini-2.5-flash-lite` (held the slot for a few hours on 2026-08-28, also
  replaced on cost at $0.10/$0.40 per MTok); `amazon/nova-micro-v1` (cheaper still, but
  at risk of a floor effect on this task); `openai/gpt-4o-mini-2024-07-18` (dated slug;
  see the correction below); `qwen/qwen3.7-flash` (cheap, 1M context, but rate-limited
  upstream when probed).

  **Correction 2026-09-03: the 128k rejection reason was inconsistent.**
  `nova-micro-v1` was recorded as rejected for "128k context and floor-effect risk" and
  `gpt-4o-mini-2024-07-18` for "dated slug, but 128k context". But the retained
  `gpt-oss-120b` has a context of 131,072 = 128k. Two arms were rejected for the same
  context length a retained arm has, so context cannot have been what separated them.
  nova-micro's **floor-effect risk** is a separate reason and still stands on its own.
  The gpt-4o-mini entry gave context as its only reason and **cannot stand as
  written**: it is out because the third slot is a commercial reference chosen on cost
  (nova-lite at $0.06/$0.24 per MTok), not because of its context window. Both arms stay
  rejected; the recorded reason is corrected, not the decision. The rejected-arms
  comment in `config.py` has the same correction.

  **What this means for the retained arm.** `gpt-oss-120b` has that same 128k limit,
  the smallest of the three arms by a factor of 8 (128k against deepseek-flash's 1M and
  nova-lite's 300k). It is therefore the arm most likely to raise
  `ContextWindowExceededError`, which section 5a item 5 already names as the most
  likely unmapped HTTP 400, and an unmapped 400 halts the grid.
- **Prompt caching:** **disabled throughout** (`config.PROMPT_CACHING = "disabled"`,
  settled 2026-08-28). Providers differ in caching policy and pricing, and the agent
  transcript is a growing prefix, so caching would add an uncontrolled provider-policy
  term to the cross-arm cost comparison, which is a secondary endpoint. Cost is
  controlled by the scope-cut levers instead.
- **Prompt strategy (4 levels):** `minimal` / `structured` / `chain_of_thought` /
  `few_shot`, written in `src/grant_eval/agent/prompts.py`. `minimal` is byte-identical
  to the string every skeleton run so far used; the other three were written on
  2026-09-01 and are new to the design.

  **The levels share a byte-identical task core and are not nested.**
  `_TASK_CORE` (role sentence, inclusion rule, snapshot-date rule, tools sentence,
  submit/empty-list rule) is the whole task definition, and it is a byte-identical
  prefix of all four rendered system prompts. `minimal` is the core and nothing else.
  Each other level is the core, a blank line, and exactly one scaffold:

  | Level | Scaffold |
  |---|---|
  | `structured` | An ordered procedure (survey links from the entry URL, open each candidate, test it against the inclusion rule, record the fields, submit) plus a restatement of the record field names. It deliberately repeats what the `submit_opportunities` tool schema already carries; that repetition is the treatment. |
  | `chain_of_thought` | Say what you are looking for before each tool call; work through the inclusion rule against the page's actual wording before recording or discarding a candidate. |
  | `few_shot` | Five worked examples. |

  `few_shot` does not also carry the `structured` field list and `chain_of_thought` does
  not carry the examples, so every pairwise contrast against `minimal` isolates a single
  change. A scaffold that also sharpened the inclusion rule would make this factor
  measure "better task spec" rather than "scaffolding", so no scaffold adds scope
  wording, field definitions or date-format rules: organisation and process only.
  Checked by tests, not just stated:
  `tests/test_prompts.py::test_task_core_is_a_byte_identical_prefix_of_every_strategy`
  and `::test_the_levels_are_orthogonal_not_nested`, with the exact bytes of all four
  pinned against golden files. All scaffolding is in the system prompt; the user prompt
  (funder, snapshot date, entry URL) is byte-identical across the four
  (`::test_user_prompt_is_invariant_across_strategies`). The four also have distinct
  fingerprints, so their cells cannot collide on `config_hash` and so on output path
  (`::test_prompt_fingerprints_are_all_distinct`).

  **`few_shot` has K = 5**: four positives and one negative, covering a dated deadline
  after the snapshot date; narrow eligibility (in scope, because the rule asks whether a
  call is open to *some* UK-based researchers); a staged multi-date call whose opening
  date is month-only (month precision kept, never an invented day, flagged in
  `uncertain_fields`); a call with no deadline that the page says is open (`deadlines`
  left empty); and an expired call still on the site (out of scope: the date decides,
  not the words "applications open"). The block is about 740 tokens on every
  completion, and with `PROMPT_CACHING = "disabled"` it is re-sent each time. That is a
  known and accepted cost of this level.

  **Keeping the examples out of the scored data.** Every example is drawn from `raeng`
  or `ihr`, the two deferred batch-3 funders in `data/ground-truth/deferred-batch3/`,
  which are in **neither split**: they are absent from `data/funders.json`, so no cell
  runs them, and `scoring/groundtruth.py` does not walk subdirectories, so none of their
  rows enters any denominator. `tests/test_prompts.py::test_few_shot_cites_no_split_funder`
  checks that the rendered `few_shot` prompt contains no slug, display name or entry-URL
  domain of any of the 25 funders in `data/funders.json`, reading that file rather than
  a hardcoded list. This test rejected the first candidate for the narrow-eligibility
  example, whose page text names the Leverhulme Trust.

  **As a result, `raeng` and `ihr` can no longer be added back to the tune split.** That
  option, offered below and in `data/ground-truth/deferred-batch3/README.md`, is
  withdrawn; the tune split stays at 16. Both CSVs now carry `split = exemplar`, a value
  `schema.Split` deliberately rejects, so moving either into `data/ground-truth/` raises
  an error at load (`schema._parse_split`) rather than quietly putting a funder whose
  pages are quoted word for word in the prompt into the scored tune split.

  **A `chain_of_thought` x model interaction is expected and pre-registered here.**
  `deepseek-flash` and `gpt-oss-120b` run at `reasoning_effort: medium` with
  `config.LOOP["carry_reasoning"] = True`, so they already produce a reasoning stream
  and the loop already carries it into later turns; an instruction to reason before
  acting is partly redundant for them. `nova-lite` has no reasoning mode, so the
  instruction has its full effect there. Predicted direction: the `chain_of_thought` -
  `minimal` gap is largest on `nova-lite`. The scaffold deliberately does not name a
  channel (it never says "output your reasoning in the following format"), because a
  channel-specific instruction would conflict with the reasoning stream the router
  already reads back.

  **Terminal state is reported by prompt strategy.** `step_budget` is 16, and
  `chain_of_thought` and `few_shot` both spend output tokens and turns before the first
  tool call, so prompt strategy can shift the terminal-state distribution
  (`step_budget_exhausted` / `abstained` / `submit_failed`) as well as extraction
  quality. Without that split, a `chain_of_thought` F1 drop would read as worse
  extraction when it may be budget exhaustion, which is a different finding with a
  different fix.
- **Temperature:** 0.0 / 0.7
- **Agent config: one level, so not a factor.** Collapsed to a single level on
  2026-08-28: `standard`, `step_budget` 16. `fetch_depth` was **deleted** from
  `AGENT_CONFIGS` and from the config hash the same day and is no longer part of the
  design. Earlier drafts described a "shallow (fetch depth 1) / deep (fetch depth 3)"
  factor, which is withdrawn. It had no effect: `tools.list_links` returns every
  on-site link by design and the loop reads only `step_budget`, so it was an unused key
  inside a pre-registered hash. Deleting it changes the hash; no grid run predates the
  change.
- **Replicates:** 5
- **Funders:** tuning split only (validation split held out)
- **Grid size:** 120 cells/funder x **16 tune funders** = **1,920 runs** on the tuning
  split, 640 per arm. Batch 3 is committed (`efe83b5a`) and recorded in `data/funders.md`,
  which is why the tune split is 16 rather than 14. Adding back the two deferred batch-3
  funders (`raeng`, `ihr`) would have made it 18 tune funders and 2,160 runs; that
  option is **withdrawn**, because they are the `few_shot` example source (above) and
  their CSVs now refuse to load into a split.
- **Cost: measured, but only just.** Earlier drafts gave a token-ceiling extrapolation
  (~$52-131, and ~£790 while Haiku held the commercial slot). Both came from a quadratic
  token model fitted to two runs at retired step budgets, both were wrong, and they
  were wrong in **both** directions. They are withdrawn rather than restated. What
  exists is a handful of single skeleton runs, all Leverhulme / `minimal` /
  temperature 0 / one Ralph iteration:

  | Run | in / out tokens | cost | latency | submitted |
  |---|---|---|---|---|
  | nova-lite, `standard` (16 steps) | 35,263 / 1,685 | $0.0025 | 14.1 s | 6, schema-valid |
  | gemini-2.5-flash-lite, `standard` (16 steps) | 152,572 / 5,928 | $0.0069 | 23.7 s | 0, schema-invalid; no longer an arm |
  | deepseek-flash, retired `shallow` (8 steps) | 52,605 / 11,553 | $0.0034 | 140.2 s | 2, schema-valid |
  | deepseek-flash, retired `shallow` (8 steps) | 117,861 / 13,468 | $0.0064 | 237.6 s | 8, schema-valid |
  | deepseek-flash, retired `deep` (24 steps) | 1,096,804 / 74,521 | $0.0297 | 572.5 s | 0, budget exhausted |

  Only the first two ran at the grid's actual config. Across all five, cost per run
  ranges from **roughly $0.0025 to $0.03**, and the spread follows the arm and how early
  the agent submits much more than the nominal budget. At that rate the full tuning
  grid costs **in the order of tens of dollars**, not the ~£790 the Haiku arm implied.
  Money is no longer the limiting factor; **wall-clock time is**. The measured
  latencies put 1,920 runs at between ~7.5 h and ~305 h serial, which is too wide to
  plan against: no run at the grid config exists for `deepseek-flash` or
  `gpt-oss-120b`. There is no longer a Ralph multiplier on top of that: section 3 set
  `max_iterations = 1` on 2026-09-01, removing the 1-5x range that was the largest
  single unknown in this estimate while the iteration count was open.

  **Any single one of these runs means very little.** The one cell run twice with a
  byte-identical config at temperature 0 scored F1 0.33 and 0.78. The 5-funder dry run
  replaces this table with an estimate worth acting on, and the figure recorded here at
  freeze must come from that dry run, not from this table.

## 2a. Task entry point: the crawl entry URL, not the homepage

`src/grant_eval/scoring/skeleton.py` used to say in a comment that "the grid always uses
the homepage" as the agent's starting point. That was wrong for most funders. The
comment was corrected in the source on 2026-08-31, but the rule is recorded here
because this file, not a comment, is where it is fixed. It is the kind of assumption
that looks obviously true from the interactive skeleton runs, which have only ever
used Leverhulme. Batch-2 and batch-3 mirrors were captured with `--no-parent` limited
to a funding-section path, not the whole site, so the homepage is missing from most
mirrors: confirmed missing from the `british-academy` and `prs-foundation` mirrors, and
present only for `leverhulme`, whose crawl happened to start at its homepage. A runner
(or a task description) that hardcoded the homepage would start most funders on a 404.

**The entry URL is the crawl entry URL**, the URL each funder's mirror was actually
crawled from, recorded per funder as `entry_url` in `data/funders.json`. This is also
the accurate description of the task the agent is given: "find the opportunities in
this funder's funding section", not "find them starting from the homepage".

Checked in code, not just stated: the `entry_url_in_mirror` check in
`grid/preflight.py` resolves every in-scope funder's `entry_url` through
`dataset/serve.py`'s own `build_lookup`/`resolve` (the same resolver the agent is
served through at run time), and a full-grid run refuses to start if any of them
returns 404.

## 2b. Execution order and the seed

Cells run in a fixed shuffle over the full 1,920-cell plan, seeded by
`config.GRID_SEED = 20260831`. Running funder by funder would confound any
time-varying provider behaviour with funder; running config by config would confound
it with config. Randomising turns provider drift across a multi-night grid into noise
rather than bias.

`GRID_SEED` is left out of both `config_hash` and `result_sha`: out of `config_hash`
because it would move all 1,920 output paths for a parameter that cannot change any
single cell's result, and out of `result_sha` because changing the seed would then
block resume. Checked by tests:
`tests/test_config_identity.py::test_grid_seed_is_not_in_the_config_hash_payload` for
`config_hash`, and `tests/test_grid_freeze.py::test_grid_seed_is_not_in_result_sha`
for `result_sha`.

## 2c. Wall-clock caps and timeout as a result

Two caps, both fixed here rather than decided during the run:

1. **Per model request: 180 seconds** (`--iteration-timeout-s`, default 180).
2. Per cell: set from the 5-funder dry run (section 5a), then fixed for the grid
   (`--cell-timeout-s`).

**Cap 1 applies per request, not per Ralph iteration.** An earlier draft of this
section fixed "per Ralph iteration: ~15 minutes", which the code does not implement and
never did: `--iteration-timeout-s` sets the per-request timeout (`TIMEOUT_S` in
`agent/router.py`, which LiteLLM applies to each completion call), and one Ralph
iteration makes up to `step_budget + 1 = 17` completions (`agent/loop.py`). The
consequence: at 180 s per request, one Ralph iteration is bounded at roughly 51 minutes
of model time, and at `max_iterations = 1` (section 3) a cell is bounded at the same
~51 minutes rather than a multiple of it. This is why the `--cell-timeout-s` default of
5400 s (90 minutes) fits the per-request cap: 90 minutes is above the worst case a cell
can reach on request timeouts alone. At the earlier `max_iterations = 5` the same sum
gave ~4.25 hours per cell, and the 90-minute cap could fire on a cell that was slow
rather than stuck. **The per-cell cap, not the per-request one, is still what actually
limits a cell**: 17 requests rarely each run to 180 s, so in practice the wall-clock
cap is the binding limit, not the sum of per-request caps. The old 900 s per-request
default would have put the per-iteration figure at ~4.25 hours. A true per-iteration
cap would have to be implemented inside `agent/ralph.py`, which is inside
`result_sha`; it is not in this build and is not pre-registered here, because
pre-registering a parameter the code does not implement is worse than naming no cap
at all.

Both caps, plus `--max-spend-usd` and `--per-arm-limit`, are recorded in the results
folder's `grid_run.json` on the first session and on every resume, and a resume with
different values is refused unless `--force-resume` is passed (in which case both sets
are recorded). A cap is not in `config_hash` or `result_sha`, but it decides which
cells come back `timed_out = True`, which is a scored result (see the next paragraph),
so a run resumed under a changed cap would otherwise mix two regimes in one dataset
without anyone noticing.

A timeout is a **result**, not a failure: it is recorded as `timed_out = True` on the
`ExtractionResult`, scored on whatever the cell had produced by then, and never
retried. It also counts as a success for the circuit breaker. Otherwise an arm that
hangs on every cell would collect a night of empty `timed_out` results without ever
tripping the breaker, and an arm that keeps timing out is itself a finding about that
arm, not an infrastructure fault to hide with a retry.

## 3. Ralph loop: one iteration (settled 2026-09-01)
Values in `config.RALPH`:
- `max_iterations`: **1**
- termination rule: `schema_valid_and_no_new_opportunities`, **which has no effect at this value**
- `scored_iteration_policy`: `final`, **which has no effect at this value**

This was the last design parameter left open. Every other design decision was settled
on 2026-08-28; this one was left for a measurement, which is now cancelled (see below).

**Why one.** Huntley's loop converges because the filesystem is shared state, so
iteration 2 reads what iteration 1 wrote and work builds up. `agent/ralph.py`
deliberately does the opposite: a fresh backend every iteration, with `messages`
rebuilt from scratch, so each iteration is an independent draw and nothing builds up.
With nothing building up there is nothing to converge *to*. The termination rule
becomes a stability test over independent samples, it fires exactly when a draw comes
back thin, and `scored_iteration_policy = "final"` then keeps that thin draw (an
8-then-2 sequence converges and the cell scores 2). Repeated independent draws of one
cell are exactly what `REPLICATES = 5` already does, and does properly (fixed n, all
five scored, variance reported for RQ3), so the outer loop was a second, less
well-specified copy of the replicate structure nested inside it, at 2-5x the
wall-clock time, which is the limiting factor.

**Kept but inactive, not deleted.** With one iteration, the termination rule has no
previous iteration to compare against and `final` simply scores the only iteration
there is. Both keys stay in `config.RALPH` and so in `config_hash`, which keeps the
hash payload's shape and keeps the history readable; `run_ralph` still reads both.
Moving `config.RALPH` from 5 to 1 changed `config_hash` for every cell, as expected; no
scored grid run predates the change.

**The calibration run is cancelled.** The earlier version of this section proposed 5
runs on Leverhulme at `max_iterations = 5` to choose between `max_iterations = 1`,
scoring the **union** across iterations, and keeping `final`. There is no longer a
policy to choose, so there is nothing for that run to measure. The per-arm timing it
would also have given is now the job of the 5-funder dry run (section 5a). `best_of`
remains dropped because it leaks information, and `union` was never implemented.

**What the loop was assumed to do, and where that goes instead.** Earlier drafts
treated iterating as a way to raise recall. It is not one. Run-to-run reliability is
answered by the 5 replicates (RQ3, as designed), and "does pooling several draws beat
one draw?" can be answered afterwards as a union over those replicates, from data
already collected, with no extra runs. The inner ReAct loop in `agent/loop.py` is
unaffected and is still what every run does.

### Loop and operational parameters
In `config.LOOP` and in the config hash.
- `budget_notice_at`: 3
- `carry_reasoning`: True. Two of the three arms now report reasoning support
  (deepseek-flash, gpt-oss-120b), so this affects most of the design rather than one
  arm; it has no effect for nova-lite, which has no reasoning mode.
- `OPENROUTER_TIMEOUT_S`: 180 s per request (default).

Two questions about what `step_budget` counts were settled on 2026-08-28.

- **The single schema-repair turn does not count against `step_budget`.**
  `step_budget` means *exploration steps*; a rejected submit gets its own repair turn
  and does not use one up. Reason: while repair counted against the budget,
  `budget_notice_at` was tuning a trade-off between recall and validity rather than
  acting as a stopping threshold. At a notice of 2 the agent explored everything and
  had no step left to repair a rejected submit; at 3 it stopped early with a thin
  answer. That is what produced the F1 0.33 / 0.78 pair. It also makes `step_budget`
  comparable across arms instead of penalising an arm that formats badly twice.
- **Reasoning-only turns do count against `step_budget`.** Every `backend.complete`
  call is one step, including a completion that produces reasoning and calls no tool.
  This is reported as a limitation: **at equal `step_budget` a reasoning arm gets fewer
  effective tool calls than a non-reasoning one**, so the budget is measured in model
  turns, not in work done. It is a deliberate choice for cost parity: counting only
  tool-calling turns would leave cost and wall-clock time unbounded for a reasoning
  arm, which is worse in a study whose secondary endpoint is a cost-effectiveness
  frontier. The hybrid (count tool calls, plus a separate completion ceiling) was
  rejected as a second operational parameter to justify for little gain.

## 4. Planned analysis

> **Zero-positive funders: settled 2026-09-13 by withdrawal (section 6a).** This was an
> open question to settle before the freeze.
> `wellbeing-of-women` had 19 labelled rows and **zero** in scope, so its recall
> denominator was 0 and its per-funder F1 was 0 or undefined for every arm, every prompt
> and every temperature, whatever the agent did. It contributed a constant, not a
> measurement. That affected two analyses below: the bootstrap CIs clustered on funder,
> and the paired per-funder non-parametric tests, where a funder that cannot vary
> either narrows the interval wrongly or drops a pair, depending on how it is handled.
> The options were to include it, to exclude zero-positive funders from the
> funder-level analyses while still reporting them in the dataset description, or to
> score it on precision and abstention alone (it was a clean abstention test, which is
> the one thing it *did* measure).
>
> **The rule adopted is the exclusion option, applied to the dataset rather than the
> analysis: the funder is withdrawn from the set entirely** (section 6a). This loses
> the abstention test, which is a real loss and is recorded as one. The question never
> came up on the tune split (every one of the 16 tune funders has at least one
> in-scope row), so no tune result depends on it, and the decision was made without
> looking at any validation page. The rule is now checked by a test:
> `tests/test_registry.py::test_every_registry_funder_has_at_least_one_in_scope_row`
> fails if any registry funder is ever labelled with zero in-scope rows again.

- Bootstrap 95% CIs on F1, **clustered on funder**.
- Paired non-parametric tests (Wilcoxon signed-rank / sign) + **Holm-Bonferroni** across
  the family of pairwise config comparisons; effect sizes alongside p-values.
- Factor effects and interactions (model / prompt / temperature).
- Reliability (RQ3): replicate consistency, temperature stability.
- Stratified by edge-case stratum, funder type, structural complexity.

### Analysis strata (settled 2026-08-28)
The ground truth is 311 rows across 25 funders, 167 of them in scope. In-scope stratum
counts: `restricted` 78 (47%), `multi_deadline` 57 (34%), `rolling` 19 (11%),
`ambiguous_eligibility` 10 (6%), `pdf_only` 3 (2%), `js_portal` 1 (1%).

Those counts are over the 25 `<slug>_<YYYY-MM-DD>.csv` files in `data/ground-truth/`.
Two other files in that folder are not ground truth and are excluded by the file-name
rule, not by inspection. Two more files, `ihr` and `raeng`, sit in
`data/ground-truth/deferred-batch3/`, match the naming rule, and are excluded by
location, because the loader does not walk subdirectories. Moving either into the main
folder is also refused at load (`schema._parse_split`): they are the `few_shot`
example source (section 2) and must stay out of both splits, and moving one would
anyway change the row counts, the in-scope total and the tune split, and so every
figure pre-registered here. Record the file set at freeze, not just the totals.

- **`restricted` is an analysis stratum, kept as a single binary tag** for RQ4's
  restricted-vs-open contrast. At 78/167 (47%) it is a near-balanced binary factor with
  good power. It is *not* re-tagged into membership / nationality / institution
  gating: that is what RQ4 would most like, but it means re-tagging 78 rows against an
  18 Sep freeze and would leave ~26 rows per sub-stratum. (An earlier suggestion to
  drop it was based on an out-of-date 65% figure and no longer applies.)
- **`pdf_only` (3 rows) and `js_portal` (1 row) are described, not analysed.** They are
  reported in the dataset description; no per-stratum F1 is computed for them, and
  this is stated where the strata results are given.
- `multi_deadline` at 57 rows (34%) is now large enough to be a stratum on the same
  grounds as `restricted`: detection needs the full set of dates to match, which is a
  real source of difficulty. Not decided; settle it before freeze.

## 5. Tune / validation split
**16 tune / 9 validation across 25 funders**, fixed and committed in `data/funders.md`
before any grid run (batch 3 committed at `efe83b5a`). Validation pages are not opened
during tuning. The best config(s) are fixed (IDs + hashes) BEFORE the single-shot
validation run, with no iteration afterwards.

The split stays at 16/9 rather than shrinking the tuning side. Cost is no longer the
limit (section 2), and config selection is the step most limited by noise: one cell
has been seen to range from F1 0.33 to 0.78 at temperature 0, and averaging over 16
funders x 5 replicates is the direct fix. 9 validation funders still support RQ6.
(This reverses earlier advice to shrink to ~8 tune funders, which was driven by the
withdrawn ~£790 cost figure.)

**Validation is 5 replicates × 9 funders = 45 runs**, the same replicate count as the
tune side, for the same reason: a 1-run validation draw would make the RQ6
tune-to-validation gap partly an artefact of comparing a mean over 5 replicates with a
single sample, in a design where one cell has already produced F1 0.33 and 0.78 from a
byte-identical config at temperature 0. **"Single-shot" is pre-registered as meaning no
config iteration after seeing validation results**, not one sample per funder. The
primary statistic is the per-funder mean over the 5 replicates; the spread across
replicates is reported as a secondary reliability result.

## 5a. The dry run

Before the grid runs, 5 funders, chosen on purpose to cover the range that drives
wall-clock time (the smallest and largest tune mirrors plus three mid-sized ones), each
run a stratified subset covering all 3 models × all 4 prompts at 1 replicate: 24 cells
per funder, **120 runs in total**. The aim is to size and time every arm, not to draw
inferences.

Dry-run cells are written to the same `config_hash` paths as grid cells and are
**reused** as grid cells. They are identical runs, and throwing away hours of work to
keep "dry run" and "grid" neatly separate is the wrong trade. This is recorded here so
that the reuse is a stated choice made in advance.

**Command.** `uv run python -m grant_eval.run --funder cara --funder
british-academy --funder carnegie-trust --funder leverhulme --funder nuffield
--replicate 1`: the five chosen funders named, with `--replicate 1` dropping the other
four replicates; every model, prompt and temperature stays in. Add `--dry-run` to print
the 120-cell plan without spending anything. Checked by a test:
`tests/test_run_cli.py::test_the_preregistered_dry_run_is_120_cells` checks the count,
so this section and the CLI cannot drift apart.

### Dry-run checklist

What the dry run has to produce before the grid is started, and what to expect while
it runs:

1. **The per-cell wall-clock cap** (section 2c), from the observed spread of per-cell
   latency. This is the only number in this file the dry run is required to settle.
2. **The per-arm timeout rate.** An arm that keeps hitting the cap is a finding about
   that arm, not an infrastructure fault (section 2c), but it is also the sign that
   the cap value needs another look before 1,920 cells run under it.
3. **The per-arm cell-failure rate after retries**, against which the circuit breaker's
   settings (`grid/policy.py`) are to be reviewed, using real rates rather than more
   modelling.
4. **The projected wall-clock time at 8 workers**, which triggers the section 5b
   scope-cut levers if it is over 96 hours.
5. **Expect an unmapped HTTP 400 to halt the grid; this is by design.**
   `grid/policy.py` treats any exception it does not recognise as HALT the first time
   it sees it, so that it gets classified on purpose rather than guessed. Status 400 is
   not in the halt-status list, so LiteLLM's whole `BadRequestError` family
   (`ContextWindowExceededError` most likely, with
   `ContentPolicyViolationError`/`UnsupportedParamsError` behind it) falls through to
   that default. A cell whose transcript grows past the model's context window will
   therefore stop the grid rather than be retried or recorded, and the run header will
   name the exception class. That is the intended behaviour of the default-to-HALT rule,
   not a bug to chase in the middle of the night. The decision it forces is whether
   that exception should become a RETRY, a RESULT or stay a HALT, which is a one-line
   change in `policy.py` (outside `result_sha`), made on purpose.
6. **Expect `arm <label> saturated` lines** in the log if one provider slows down
   badly. They mean cells were put back in the pending set so the pool could work on
   other arms; the next resume re-runs those cells unchanged. A steady stream of them
   is a throughput signal, not a data-integrity one.
7. **Count `saturated` log lines during the dry run, and raise
   `ARM_SATURATION_TIMEOUT_S` (`grid/executor.py`, default 120.0) if they are common
   under a healthy provider.** Two separate measurements (real threads time-scaled to
   production, and an event-driven simulation at full grid scale) found that the
   original 5.0 s value fired on ordinary contention on a healthy arm. Cell latency
   runs to tens or hundreds of seconds, 20-100x that timeout, so it could not tell a
   busy arm from a dead one; it deferred 8-15% of cells on the first pass and took 3-4
   passes to clear them. `saturated` lines under a provider that is not actually
   struggling are the same signal at 120.0's margin; set the constant from the
   observed p50 cell latency rather than trusting either number blindly. It is an
   operational setting, not part of `result_sha`, and can be changed mid-grid.

## 5b. Scope-cut levers: order and trigger

**Trigger (settled):** if the dry run (section 5a) projects more than 96 hours of
wall-clock time at 8 workers, the levers are applied in order until the projection is
under 96 hours.

**Lever order: settled 2026-09-04.** Replicates are never cut. The order below
replaces the one previously recorded in this section, whose first lever (replicates
5 -> 3) contradicted four other parts of the design, including section 5 above:
config selection is the step most limited by noise, and averaging over 16 funders x 5
replicates is the direct fix for it. A lever that removes the fix for the design's
largest known source of noise cannot be authorised in advance.

The earlier lever order could not simply be reused: its first two levers ("drop the
`deep` agent config", "drop the Haiku arm") no longer exist, because the agent-config
factor was removed and the commercial-reference arm was swapped to nova-lite on
2026-08-28. The new order is therefore:

1. temperature 2 levels -> 1 (keep 0.0)
2. tune funders 16 -> 12
3. drop one prompt strategy (drop `chain_of_thought` first: lowest mean F1 in the dry run)

**Replicates (5), the separate validation run and the kappa pass are never cut.**

**In practice the trigger does not fire.** The section 5a dry run (120 cells,
2026-09-03/04) measured the full grid at ~18.5 hours of wall-clock time at
`--per-arm-limit 4` and ~$22, against a 96-hour trigger. The levers are recorded for
completeness, not because they are expected to be used.

## 6. Freeze record

**Frozen 2026-09-04.** The configuration below is the pre-registered configuration. Any
change after this point is an amendment to the pre-registration and must be recorded
as one.

- **Timestamp:** 2026-09-04
- **Git SHA at freeze:** `9354b155575a0e882e580dfa6082929340bb8ba6`
- **`result_sha` at freeze:** `fa4ce518461bf65f51f9baa08eefb4dfe7d72a05550d8229b073e12dc636b1d0`

**Design (factor levels and fixed parameters):** 16 tune funders x 4 prompt strategies
x 2 temperatures x 3 models x 5 replicates = 1,920 cells. `agent_config` has one level
(`standard`, `step_budget: 24`). `RALPH.max_iterations: 1`;
`LOOP.budget_notice_at: 3`, `LOOP.carry_reasoning: True`; `PROMPT_CACHING: disabled`;
`TEMPERATURES: [0.0, 0.7]`; `REPLICATES: 5`.

**Prompt-template fingerprints** (SHA-256 of the rendered template with fixed
placeholder values; these are what `config_hash` includes):

| strategy | fingerprint |
|---|---|
| minimal | `719a1023d3074c2a9c8cc5018b7335dedfb9272dbdc4a8373d145e6bdb6e31b9` |
| structured | `6821a37196845c02a793f4286e4d36e330206bd8c20f1115228696194a1e759b` |
| chain_of_thought | `2fff49924ddefbfeb78f363f774b17402176164b76b0b62f025084fdaf0ae4d6` |
| few_shot | `ed69dcf5cd7007471bd51b277fe3bde6e887a3a00d964ff3194a2eb1dd9681fb` |

**Reference cell hashes** (prompt `minimal`, t=0.0, `standard`), as a check on the
frozen config: recomputing these on a later checkout detects any drift.

| arm | config_hash |
|---|---|
| deepseek-flash | `6193098e749b172a7fa3120654fcb230dd99c9da9d666e75ad1ee266c5855472` |
| gpt-oss-120b | `e0e1ed45b34735e97dc3283761a5ffc8427eb766bc2ab736956679b9c50d3127` |
| nova-lite | `6e47d74744df8caf99319462a9a78c9a7952d429cc898e612214b0b5a6b657b4` |

**Model snapshot IDs, provider, quantization, `reasoning_effort`** (all
`allow_fallbacks: False`):

| arm | slug | provider | quantization | reasoning_effort |
|---|---|---|---|---|
| deepseek-flash | `openrouter/deepseek/deepseek-v4-flash-0731` | parasail | fp8 | medium |
| gpt-oss-120b | `openrouter/openai/gpt-oss-120b` | deepinfra | bf16 | medium |
| nova-lite | `openrouter/amazon/nova-lite-v1` | amazon-bedrock | n/a (closed weights) | none |

**Served provider read off a live response, per arm.** Checked over the 120-cell dry
run of 2026-09-03/04: every cell's served `provider` was read off the live response and
matched the pin (Parasail 40/40, DeepInfra 40/40, Amazon Bedrock 40/40). Served
quantization cannot be checked: nothing reports it, and `quantizations` is only a
request-side filter (see section 2). `assert_provider_pin` halts the grid on a provider
mismatch.

**OpenRouter catalogue re-check date:** 2026-09-03. All three endpoints were probed live
for availability, quantization and context length. Recorded context limits:
deepseek/parasail 1,048,576; gpt-oss/deepinfra 131,072; nova-lite/bedrock 300,000.

**Ground-truth file set at freeze** (25 files; 351 rows, 174 in scope, but see the
reconciliation note in section 6a: the loader reports 311/167 over the same 25 files
and the two figures have never been reconciled). `_TEMPLATE.csv`,
`ground-truth_all-funders (rsc).csv` and
`ground-truth_all-funders(british-ecological-society).csv` are excluded by
`scoring/groundtruth.py` and are not part of the set. **One file in the list below,
`wellbeing-of-women_2026-08-21.csv`, was withdrawn after the freeze (section 6a).** The
list is left as it was at freeze, because a freeze record edited to match later
decisions is no longer a freeze record:

```
british-academy_2026-08-28.csv          nuffield_2026-08-16.csv
british-ecological-society_2026-08-16.csv  pathological-society_2026-08-28.csv
cara_2026-08-21.csv                     paul-mellon_2026-08-16.csv
carnegie-trust_2026-08-16.csv           physiological-society_2026-08-21.csv
fulbright_2026-08-16.csv                pilgrim-trust_2026-08-21.csv
garfield-weston_2026-08-16.csv          prs-foundation_2026-08-28.csv
genetics-society_2026-08-28.csv         royal-hist-soc_2026-08-16.csv
hakluyt_2026-08-21.csv                  rsc_2026-08-21.csv
henry-moore_2026-08-14.csv              sasakawa_2026-08-21.csv
heritage-fund_2026-08-16.csv            society-antiquaries_2026-08-21.csv
leverhulme_2026-08-09.csv               society-of-authors_2026-08-21.csv
nuffield-farming_2026-08-21.csv         wellbeing-of-women_2026-08-21.csv
                                        wolfson_2026-08-16.csv
```

**Supervisor sign-off:** Eugene Kozlovski signed off the ground-truth schema and the
tune/validation split, as reported on 2026-09-03.

## 6a. Amendment 2026-09-13: `wellbeing-of-women` withdrawn from the validation split

**This is an amendment made after the freeze and is recorded as one.** The freeze
record in section 6 is left exactly as it was; this section states what changed after
it.

**Change.** `wellbeing-of-women` is removed from the funder set. The set goes from 25
funders to **24**, and the split from 16 tune / 9 validation to **16 tune / 8
validation**. The tune split is unchanged.

**Reason.** 19 labelled rows, **0 in scope**: every scheme was closed on the
2026-08-21 snapshot date. Its recall denominator is 0, so its per-funder F1 is a
constant for every arm, prompt and temperature. Under funder-clustered bootstrap CIs
and paired per-funder tests it adds a constant rather than a measurement, which either
narrows intervals wrongly or silently drops pairs. Section 4 listed this as an open
question to settle before the freeze and it was never settled; it is settled here, by
withdrawal.

**Why this is not a decision made after seeing the data it affects.**

1. `wellbeing-of-women` is a **validation** funder. Its pages have never been looked at
   and no run has ever been made against it, so no validation result informed this.
2. The question does **not arise on the tune split** (every one of the 16 tune funders
   has at least one in-scope row), so no tune result, and nothing in
   `eval/figures/report.txt`, changes by a single digit. The tune leaderboard, the
   confirmatory contrasts, the strata analysis and the cost frontier are all
   unaffected.
3. The zero-in-scope fact was recorded on **2026-08-26** in the first-pass label
   corrections, before the freeze, and `pathological-society` had already been added in
   batch 3 as its replacement (`data/funders.md`, row 23). The finding is old; only
   what is done about it is new.

**What this costs.** Section 4 noted that a zero-positive funder is a *clean
abstention test*, the one thing it does measure. Withdrawing it gives that up.
Abstention is still reported on the tune split (`ABSTENTION` in the analysis output,
216 abstained runs of 1,920), so the measure does not disappear, but the deliberate
zero-positive probe does. This is accepted, not worked around.

**How.** The registry entry is deleted from `data/funders.json`, so the funder is never
listed as a grid cell. `data/ground-truth/wellbeing-of-women_2026-08-21.csv` is
**moved, not deleted**, to `data/ground-truth/withdrawn/`, and excluded by location: the
same rule (the loader does not walk subdirectories) that keeps the `few_shot` example
funders in `deferred-batch3/` out of both splits. The labelling work stays available to
check.

**Resulting counts.**

| | before | after |
|---|---|---|
| funders | 25 | 24 |
| split | 16 tune / 9 validation | 16 tune / 8 validation |
| ground-truth files loaded | 25 | 24 |
| candidate rows loaded | 311 | 292 |
| **in-scope rows loaded** | **167** | **167** (unchanged: all 19 withdrawn rows were out of scope) |
| validation cells listed | 1,080 | 960 |
| validation single-shot run, per chosen config | 45 (9 × 5) | **40 (8 × 5)** |

**The rule is now checked by a test.**
`tests/test_registry.py::test_every_registry_funder_has_at_least_one_in_scope_row`
fails if any registry funder is ever labelled with zero in-scope rows again. It tests
the rule, not the removal, so it also catches a future funder that drops to zero in
scope as its calls close.

**Reconciliation note, unresolved.** Section 6 records the frozen file set as "25
files; 351 rows, 174 in scope". `scoring/groundtruth.py` loads those same 25 files and
reports **311 rows, 167 in scope**, and 311/167 is the figure used everywhere else
(`data/funders.md`, the module docstring and the test suite). The 351/174 pair matches
nothing: it is not the loaded set, and it is not the loaded set plus `deferred-batch3/`
either (that would be 378/187). **This discrepancy predates this amendment and is not
resolved by it.** It must be reconciled before the dataset description is written,
and whichever figure is wrong should be corrected with its own amendment rather than
quietly overwritten here.
