# Grant discovery benchmark

An LLM agent that browses saved copies of UK research-funder websites and lists the
funding calls that are open (or about to open) on the date each copy was taken. The
agent is run across a grid of models, prompts and temperatures, and its output is
scored against hand-labelled ground truth for 24 funders. A second model acts as a
judge on the free-text fields, checking each summary for groundedness (is it supported
by the page?) and completeness (does it cover the labelled facts?).

## Folder layout

| Path | What it holds |
|---|---|
| `src/grant_eval/agent/` | The agent loop, tools, prompts and the OpenRouter client |
| `src/grant_eval/dataset/` | Snapshot capture, the local mirror servers, ground-truth workbook tools |
| `src/grant_eval/grid/` | Grid runner support: cell plan, checks before a run, logging, status |
| `src/grant_eval/scoring/` | Scorer, aggregation, frozen scores, the LLM judge and judge-human agreement |
| `src/grant_eval/analysis/` | Statistics, tables and figures |
| `tests/` | pytest suite |
| `notebooks/` | `analysis` and `judge` for the tune grid, `validation_analysis` and `validation_judge` for the validation run |
| `scripts/build_funders_json.py` | Builds `data/funders.json` from `data/funders.md` and the snapshots |
| `data/funders.json` | Funder registry read by the code: slug, split, snapshot date, entry URL, content hash |
| `data/funders.md` | The funder list and the tune/validation split, with notes |
| `data/ground-truth/` | One labelled CSV per funder, plus the review workbook. `deferred-batch3/` holds two funders kept out of both splits (they supply the few-shot examples) |
| `data/snapshots/` | Saved funder pages, one folder per funder and date |
| `data/checks/` | Human checks on the judge and on the labels (see below) |
| `eval/preregistration.md` | The analysis plan, fixed before the final runs |
| `eval/final_grid/` | Agent output for the 1,920 tune runs, with the run record and log |
| `eval/final_grid_traces/` | Step-by-step traces for those runs |
| `eval/locked-config.json` | The configs chosen on tune F1 for the validation run |
| `eval/validation/` | Agent output for the validation run |
| `eval/validation_traces/` | Traces for the validation run |
| `eval/scored/` | Frozen scores for the tune grid (`grid/`) and validation run (`validation/`) |
| `eval/judge_results/` | Judge verdicts and roll-up for the tune grid, with `judge_pass.log` |
| `eval/judge_results_validation/` | Judge verdicts and roll-up for the validation run |
| `eval/figures/` | Tables, `results.json` and figures for the tune grid |
| `eval/figures-validation/` | The same for the validation run |

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. From the project root run `uv sync`. This creates the virtual environment from `uv.lock`.
3. Copy `.env.example` to `.env` and fill in `OPENROUTER_API_KEY`. The key is only
   needed for the steps marked **paid** below. Everything else runs free from the saved
   results.

Run every command from the project root, since the default paths are relative to it.
Add `--help` to any command to see its options, except `skeleton` and `grid`, which
take no options. `skeleton` ignores `--help` and makes a live, paid call (see step 4).

## Ground truth

Each funder's CSV was labelled by hand from the saved copy of its site, never the live
site. Snapshots were taken between 2026-08-09 and 2026-08-28, and the date is in each
file name.

- A row is in scope if, on the snapshot date, it is a distinct call that is accepting
  applications (or states a future opening date) and is open to some UK-based
  researchers. Closed, expired and information-only pages are out. A call with no
  deadline counts only if the page says it is open now.
- Out-of-scope candidates stay in the file with `in_scope = no` and a reason, so they
  act as distractors.
- Fields labelled: title, canonical URL, source page, status (`open`, `forthcoming`,
  `rolling`), deadlines, next open date, and where given, amounts, eligibility
  summary, summary and discipline tags. Edge cases are tagged (`rolling`,
  `multi_deadline`, `restricted`, `pdf_only`, `js_portal`, `ambiguous_eligibility`).
- A call with several stages is one row with every date listed. A scheme with a future
  closing date but no stated opening date is out of scope.
- The scorer matches on funder and normalised URL, using the title only when two rows
  of one funder share a URL.

## How to run

Steps marked **paid** call a model through OpenRouter and need the API key. The rest
are free.

### 1. Snapshots

```
uv run snapshot --funder leverhulme --entry https://www.leverhulme.ac.uk/ --no-parent
uv run refine topup ...      # add missed pages to a snapshot
uv run refine prune ...      # drop files the study never reads
```

`data/snapshots/` here holds only the pages the study used (pages that were labelled,
fetched by the agent or read by the judge, plus each snapshot's manifest and metadata),
not the full mirrors. Re-running `snapshot` would capture today's site, not the copies
used.

### 2. Local mirror server

```
uv run serve --snapshot data/snapshots/leverhulme/2026-08-09   # one funder
uv run serve_all                                                # every funder, with a hub page on :8100
```

### 3. Ground-truth workbook

```
uv run gt_workbook     # build or extend the review workbook from the CSVs
uv run gt_export       # write the workbook back to the per-funder CSVs
```

### 4. Single test run (paid)

```
uv run skeleton
```

Runs one Leverhulme cell end to end. **Warning:** this command has no `--help`, and
running it in any form makes a live, paid call if `OPENROUTER_API_KEY` is set. Use
`SKELETON_FORCE_MOCK=1 uv run skeleton` to run the offline mock instead.

### 5. The grid (paid)

```
uv run grid                       # print the number of cells per funder
uv run grid_run --dry-run         # print the 1,920-cell plan, no calls
uv run grid_run --workers 8 --per-arm-limit 4 \
    --results-root eval/final_grid --traces-root eval/final_grid_traces
uv run grid_status --results-root eval/final_grid
uv run lock_config                # pick the best tune config(s) on F1, writes eval/locked-config.json
uv run grid_run --split validation \
    --results-root eval/validation --traces-root eval/validation_traces
```

Re-running the same `grid_run` command resumes it. The runner checks the git state
before a run, so it needs a git checkout.

### 6. Scoring (free)

```
uv run score eval/final_grid/grid --quiet-records
uv run aggregate eval/final_grid/grid
uv run freeze_scores eval/final_grid/grid                                     # writes eval/scored/grid
uv run freeze_scores eval/validation/grid --out-dir eval/scored/validation
```

### 7. Judge (paid)

```
uv run judge eval/final_grid/grid --results-root eval/judge_results --dry-run   # cost estimate, no calls
uv run judge eval/final_grid/grid --results-root eval/judge_results
uv run judge eval/validation/grid --results-root eval/judge_results_validation
```

The judge reads the agent output and the saved pages. Verdicts are cached under the
results root, so a re-run over the saved results reads from the cache.

### 8. Figures and notebooks (free)

```
uv run figures                                                    # writes eval/figures
uv run figures --scored-dir eval/scored/validation --out-dir eval/figures-validation
uv run jupyter lab                                                # then open notebooks/
```

### 9. Tests (free)

```
uv run pytest
```

Tests that need a git checkout are skipped when run outside one.

### Free path

Without an API key you can reproduce every reported number from the saved results:
steps 6 and 8, the tests, and the judge-human agreement below.

The judge-human agreement is worked out in `notebooks/judge.ipynb` from
`data/checks/judge_check.csv`. It does not call a model. The judge and the human agree
on 186 of 200 claims (93%): 91 of 100 for groundedness and 95 of 100 for completeness.

## Checks

- `data/checks/judge_check.csv`: a human review of 200 judge verdicts. Each row has
  the claim, the evidence, the judge's verdict (`judge_verdict`) and the human
  verdict (`human_verdict`), both yes/no.
- Second labelling pass: 60 ground-truth rows for carnegie-trust, fulbright, cara,
  british-academy and british-ecological-society were labelled again. In the
  ground-truth CSVs these rows have `pass_id` B and labeller "Finn Rea - label 2".
