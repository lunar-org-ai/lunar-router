# PLAN — P15.3.5 · Augmentation: judge via Agent SDK + goldens + preference data

| Field | Value |
|---|---|
| Phase | P15.3.5 |
| Parent | P15.3 (Router — UniRoute, autonomous training) |
| Status | Not started |
| Depends on | P15.3.1 (core + models), P15.3.4 (`PromptDataset`, traces), **harness brain transport** (already exists in `harness/introspection/agent.py`; P7.7 Agent SDK upgrade is additive) |
| Unblocks | P15.3.6 (evaluation can use augmented goldens), P15.3.7 (router_proposer pulls preference pairs) |
| Reference | `/Users/diogovieira/Developer/open_project/OpenTracy/opentracy/augmentation` |

## Goal

Turn unlabeled production traces into **labeled preference pairs** via
LLM-as-judge — using **the same brain transport the harness already uses**
(no second cerebro). Feed those pairs into a `PreferenceDataset` that
P15.3.6 / P15.3.7 consume to refine Ψ.

This phase ships **the judge + storage**. It does **not** run the judge
inside `/run` (still pure offline batch), it does **not** decide *when* to
judge (P15.3.9), and it does **not** consume the resulting dataset to
update Ψ (P15.3.6 / P15.3.7).

## Scope

### In scope
- `harness/brain/__init__.py`, `harness/brain/transport.py` — extract `select_transport()` + `complete(prompt, system_prompt) -> str` from the existing dual-transport logic in `harness/introspection/agent.py`. Tool-free; pure completion. Anthropic API path when `ANTHROPIC_API_KEY` is set, `claude --print` CLI fallback otherwise. Raises `BrainNotAvailableError` when neither is reachable.
- `router/augmentation/__init__.py` — package marker.
- `router/augmentation/judge.py` — port `JudgeVerdict`, `PointwiseScore`, `LLMJudge`. **Rewire** the constructor: instead of taking an `LLMClient`, take a `complete` callable from `harness.brain.transport`. Cap `compare_batch` / `rate_batch` at `max_augmentation_samples=500` (configurable, default per roadmap).
- `router/augmentation/judge_prompt.md` — extract the `PAIRWISE_PROMPT` and `POINTWISE_PROMPT` strings into a markdown file with `---` separator. Loaded at module init.
- `router/augmentation/preference.py` — port `PreferencePair` + `PreferenceDataset` verbatim. Adjust imports.
- `router/augmentation/goldens.py` — port `AugmentedSample` + `GoldenAugmenter`. Adapt: input is a list of `(prompt, model_id, response)` triples (sourced from `runtime/store/traces.py` + the existing eval suite goldens), output is a list of `AugmentedSample` plus a side-effect write to `evals/preference_pairs/<run_id>.jsonl`.
- `evals/preference_pairs/` — new directory for persisted preference data, one JSONL per cycle. Filename: `pp_<iso_date>_<short_hash>.jsonl` (matching the existing eval reports naming convention — verify during T6).
- `router/tests/test_judge.py`, `test_preference.py`, `test_goldens.py`, `test_brain_transport.py`.

### Out of scope (deferred)
- Refactoring `harness/introspection/agent.py` to use the extracted `harness/brain/transport.py`. Existing introspection keeps its private functions; only **new** code uses the new helper. (Touching working code without need violates the "Don't add features beyond what the task requires" rule.)
- Self-consistency / multi-judge ensembling. Single-judge for v1.
- Pointwise-only flow without a comparator pair. We persist both `JudgeVerdict` (pairwise) and `PointwiseScore`, but the active path is pairwise — pointwise stays a backup for prompts without a baseline response.
- Live cost accounting per judge call. The transport returns the response; cost tracking lands when token-usage accounting lands across the project.
- Fine-tuning the judge prompt with project-specific examples (will iterate after P15.3.7's first end-to-end run reveals failure modes).

## Reference → target file map

| Reference | Target | Port mode |
|---|---|---|
| `augmentation/judge.py` (`JudgeVerdict`, `PointwiseScore`, `LLMJudge`, parsers) | `router/augmentation/judge.py` | partial — port classes; **rewire** constructor to accept `complete` callable instead of `LLMClient`; extract prompts into `judge_prompt.md` |
| `augmentation/preference_data.py` (`PreferencePair`, `PreferenceDataset`) | `router/augmentation/preference.py` | verbatim — adjust imports |
| `augmentation/golden_augmenter.py` (`AugmentedSample`, `GoldenAugmenter`) | `router/augmentation/goldens.py` | partial — port classes; replace ClickHouse cache hooks with file-based persistence; default output to `evals/preference_pairs/` |
| `harness/introspection/agent.py` (`_select_transport`, `_call_claude_code_cli`, `_call_anthropic_api`) | `harness/brain/transport.py` | extract — duplicate the transport logic into a tool-free `complete()` helper. Original file untouched. |

## Pre-work

- Confirm a brain transport is reachable: `ANTHROPIC_API_KEY` env var **or** `claude` binary on PATH. The smoke test in T9 fails loudly if neither.
- No new pyproject deps (`anthropic` already core; CLI invocation uses subprocess).

## Tasks (atomic, ordered)

### T1 — Extract `harness/brain/transport.py`
Create `harness/brain/__init__.py` and `harness/brain/transport.py`:

```python
"""Shared LLM completion transport for harness sub-systems.

Mirrors the transport-selection logic in harness/introspection/agent.py
but exposes a tool-free completion API instead of the introspection-aware
loop. Used by P15.3.5 (judge) and any future harness component that
needs a plain "prompt in → text out" call against the same brain.
"""

import os
import shutil
import subprocess
from typing import Optional


class BrainNotAvailableError(RuntimeError):
    """Raised when no transport is configured (no ANTHROPIC_API_KEY, no claude CLI)."""


def select_transport() -> str:
    """Same logic as harness.introspection.agent._select_transport.
    Returns 'anthropic_api' | 'claude_code_cli' | 'none'.
    """
    forced = os.getenv("BRAIN_TRANSPORT", "").strip()
    if forced in ("anthropic_api", "claude_code_cli"):
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic_api"
    if shutil.which("claude"):
        return "claude_code_cli"
    return "none"


def complete(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    transport: Optional[str] = None,
) -> str:
    """Run a single tool-free completion against the harness brain.

    transport=None → auto-select via select_transport().
    Raises BrainNotAvailableError if no transport is reachable.
    """
    chosen = transport or select_transport()
    if chosen == "none":
        raise BrainNotAvailableError(
            "no brain transport — set ANTHROPIC_API_KEY or install `claude` CLI"
        )
    if chosen == "anthropic_api":
        return _complete_via_api(prompt, system_prompt, model, max_tokens, temperature)
    return _complete_via_cli(prompt, system_prompt, model, max_tokens, temperature)


def _complete_via_api(prompt, system_prompt, model, max_tokens, temperature) -> str:
    """anthropic SDK call. Mirrors harness/introspection/agent._call_anthropic_api,
    minus the tool-use loop (we want plain completion)."""

def _complete_via_cli(prompt, system_prompt, model, max_tokens, temperature) -> str:
    """`claude --print` subprocess. Stateless. Mirrors
    harness/introspection/agent._call_claude_code_cli body, minus history
    and tool plumbing."""
```

The file is ~80 lines. No tests for the API/CLI paths in unit tests (those need network or a real `claude` binary); behavior is exercised by the judge smoke test in T9.

Add `BRAIN_TRANSPORT` to the existing env-var documentation (likely `README.md` or `_env.py` — verify during T1).

### T2 — Port `judge.py` and rewire to `complete`
Copy `<REF>/augmentation/judge.py` → `router/augmentation/judge.py`. Then:

- Change `LLMJudge.__init__`:
  ```python
  def __init__(
      self,
      complete_fn: Callable[..., str] = complete,   # from harness.brain.transport
      *,
      max_samples: int = 500,
      judge_model: Optional[str] = None,            # passes through to transport
      pairwise_template_path: Path = Path(__file__).parent / "judge_prompt.md",
  ):
  ```
- Replace every `self.judge_client.generate(...)` call with `self.complete_fn(prompt, system_prompt=..., model=self.judge_model, temperature=0.0)`.
- Move the `PAIRWISE_PROMPT` and `POINTWISE_PROMPT` strings out of the file (T5 holds them).
- Enforce `max_samples` in `compare_batch` and `rate_batch`: if `len(inputs) > self.max_samples`, log a warning and slice `inputs[: self.max_samples]`.
- Keep `_parse_pairwise` and `_parse_pointwise` parsers verbatim. They're regex-based and brittle — note the risk in §Risks below.

### T3 — Port `preference.py` verbatim
Copy `<REF>/augmentation/preference_data.py` → `router/augmentation/preference.py`. Adjust imports (`from .judge import JudgeVerdict, PointwiseScore`). No behavior changes.

Surface (recap):
- `PreferencePair(prompt, chosen_model, rejected_model, chosen_response, rejected_response, source, confidence)`
- `PreferenceDataset(pairs, name)` — `__len__`, `__iter__`, `add`, `add_from_verdicts`, `add_from_pointwise_scores`, `model_win_rates`, `filter_by_source`, `filter_by_confidence`, `save`, `load`.

### T4 — Port `goldens.py` (slim)
Copy `<REF>/augmentation/golden_augmenter.py` → `router/augmentation/goldens.py`. Then:

- Replace any ClickHouse cache hooks (`augment_from_cache`) with a file-based variant that reads `runtime/store/traces.py:query_traces(...)` results.
- Default output path: `evals/preference_pairs/pp_<utc_date>_<short_hash>.jsonl`. Filename hash derived from `(prompt_count, judge_model, timestamp)` so reruns don't collide.
- Surface stays:
  - `AugmentedSample(golden_prompt, judge_verdicts, pointwise_scores, ground_truth_label?)`
  - `GoldenAugmenter(judge: LLMJudge, max_samples: int = 500).augment(goldens, candidate_responses_by_model) -> list[AugmentedSample]`
  - `GoldenAugmenter.augment(...)` writes the resulting `PreferenceDataset` to disk before returning.

### T5 — Extract prompts to `judge_prompt.md`
Create `router/augmentation/judge_prompt.md`:

```markdown
# Pairwise

You are an expert judge comparing two AI responses to the same prompt.

**User Prompt:**
{prompt}

**Response A** (from {model_a}):
{response_a}

**Response B** (from {model_b}):
{response_b}

Compare both responses on: accuracy, completeness, clarity, and helpfulness.

Reply with EXACTLY one line in this format:
WINNER: A|B|TIE
CONFIDENCE: 1-5
REASON: <one sentence>

---

# Pointwise

You are an expert evaluator rating an AI response.

**User Prompt:**
{prompt}

**Response** (from {model_id}):
{response}

Rate this response on a scale of 1-5:
1 = Incorrect, unhelpful, or harmful
2 = Partially correct but major issues
3 = Acceptable but could be better
4 = Good, mostly correct and helpful
5 = Excellent, accurate and comprehensive

Reply with EXACTLY one line in this format:
SCORE: 1-5
REASON: <one sentence>
```

`judge.py` loads this file once at module init, splits on `---`, and renders via `.format(...)`.

### T6 — Persistence path: `evals/preference_pairs/`
Create the directory with a `.gitignore` allowing `*.jsonl` but ignoring the index `.duckdb` shadow files (mirror what `evals/reports/` does — verify during T6 by `cat evals/reports/.gitignore`).

`PreferenceDataset.save(path)` already writes JSONL per the reference. Confirm the format during T3:
```jsonl
{"prompt": "...", "chosen_model": "...", "rejected_model": "...", "chosen_response": "...", "rejected_response": "...", "source": "judge_pairwise", "confidence": 4}
```

`GoldenAugmenter.augment(...)` writes to `evals/preference_pairs/pp_<utc_date>_<short_hash>.jsonl` and returns the path alongside the in-memory `PreferenceDataset`.

### T7 — `max_augmentation_samples` enforcement
Two places enforce the cap:
- `LLMJudge.compare_batch` / `rate_batch` — slice input if oversized.
- `GoldenAugmenter.augment(max_samples=...)` — same slice, but applies *before* the LLM calls (so we don't waste tokens then truncate).

Default everywhere: 500. Constructor-level override on both classes.

### T8 — Tests (offline)

Create `router/tests/test_brain_transport.py`:
- `test_select_transport_none` — clear `ANTHROPIC_API_KEY`, monkeypatch `shutil.which` to return None → returns `"none"`.
- `test_select_transport_api` — set `ANTHROPIC_API_KEY=dummy`, returns `"anthropic_api"`.
- `test_select_transport_cli` — clear API key, mock `claude` on PATH, returns `"claude_code_cli"`.
- `test_complete_raises_on_none` — `complete("hi")` with no transport → `BrainNotAvailableError`.
- (No happy-path tests here — those need network/CLI; happens in T9 smoke.)

Create `router/tests/test_judge.py`:
- `test_judge_compare_uses_complete_fn` — pass a `FakeComplete` that returns a canned `"WINNER: A\nCONFIDENCE: 4\nREASON: clearer"` string, assert `JudgeVerdict.winner == "A"`, `confidence == 4`.
- `test_judge_compare_batch_caps_at_max_samples` — pass 600 inputs, max=500, assert only 500 outputs + a warning was logged.
- `test_judge_pairwise_parser_handles_tie` — feed `"WINNER: TIE\nCONFIDENCE: 3\nREASON: equivalent"`, assert `winner == "TIE"`.
- `test_judge_pairwise_parser_handles_garbage` — feed `"i don't know"`, assert `JudgeVerdict` has `winner == None` and surfaces a parse-error reason (graceful, no crash).
- `test_judge_pointwise_parser_basic` — `"SCORE: 4\nREASON: solid"` → `PointwiseScore(score=4, ...)`.
- `test_judge_loads_prompt_template` — `LLMJudge` has both `PAIRWISE_PROMPT` and `POINTWISE_PROMPT` populated from `judge_prompt.md` after init.

Create `router/tests/test_preference.py`:
- `test_preference_pair_basic` — construct, attribute access works.
- `test_preference_dataset_add_from_verdicts` — feed 5 `JudgeVerdict`s with mixed winners → 4 pairs added (TIEs skip), `len(ds) == 4`.
- `test_preference_dataset_save_load_round_trip` — write JSONL → read → assert equality.
- `test_model_win_rates` — synthetic 10 pairs across 3 models → win rates sum sensibly.

Create `router/tests/test_goldens.py`:
- `test_golden_augmenter_smoke_with_fake_judge` — build a `FakeLLMJudge` whose `compare()` returns canned verdicts, run `augment()` on 5 prompts × 3 models → 5 `AugmentedSample`s + a written JSONL file under a `tmp_path`-redirected output dir.
- `test_golden_augmenter_caps_at_max_samples` — input 600 prompts, max=100, file has 100 entries.

### T9 — Smoke test (manual; requires brain)
Document in `router/tests/test_judge_smoke.py` (gated by `OPENTRACY_RUN_SMOKE=1`):
```python
@pytest.mark.skipif(not os.getenv("OPENTRACY_RUN_SMOKE"), reason="needs live brain")
def test_judge_5_real_pairs():
    judge = LLMJudge(complete_fn=complete, max_samples=5)
    verdicts = judge.compare_batch([
        ("What is 2+2?", "4", "It is four.", "ans-a", "ans-b"),
        # ... 4 more
    ])
    assert len(verdicts) == 5
    assert all(v.winner in {"A", "B", "TIE", None} for v in verdicts)
```
Run once with `OPENTRACY_RUN_SMOKE=1 python -m pytest router/tests/test_judge_smoke.py -v` to confirm the live path works end-to-end. Capture rationales in the test output. This satisfies the roadmap DoD ("smoke test: judge 5 trace pairs end-to-end, capture rationale, persist…").

### T10 — Validate
```
cd /Users/diogovieira/Developer/opentracy_new_mode
python -m pytest router/tests/test_brain_transport.py router/tests/test_judge.py router/tests/test_preference.py router/tests/test_goldens.py -v
OPENTRACY_RUN_SMOKE=1 python -m pytest router/tests/test_judge_smoke.py -v   # once
python -m pytest router/tests/ -v   # full router test surface still green
```

## Acceptance criteria (DoD)

1. `python -m pytest router/tests/test_brain_transport.py router/tests/test_judge.py router/tests/test_preference.py router/tests/test_goldens.py -v` is green.
2. `OPENTRACY_RUN_SMOKE=1 python -m pytest router/tests/test_judge_smoke.py -v` runs successfully against the live brain (`ANTHROPIC_API_KEY` set OR `claude` on PATH). 5 verdicts captured, rationales non-empty.
3. After the smoke test, `evals/preference_pairs/pp_<date>_<hash>.jsonl` exists with 4–5 lines (TIEs skip), each line a valid `PreferencePair` JSON.
4. `LLMJudge(complete_fn=complete)` raises `BrainNotAvailableError` cleanly when neither `ANTHROPIC_API_KEY` nor `claude` is reachable. (Smoke-tested by manually unsetting the env var + masking PATH; no automated test since both states are needed.)
5. `LLMJudge.compare_batch(inputs)` with `len(inputs) > 500` slices to 500 and logs a `WARNING` mentioning the cap.
6. `harness/introspection/agent.py` is **untouched** (no regression introduced).
7. `from router.augmentation.judge import LLMJudge` and `from router.augmentation.preference import PreferenceDataset` work without runtime errors.
8. No regressions: full `python -m pytest` green.

## Risks / open questions

- **Parser brittleness.** `_parse_pairwise` is regex-based: `"WINNER: A|B|TIE\nCONFIDENCE: 1-5\nREASON: ..."`. A real Claude/Anthropic completion will *usually* obey, but small format drift breaks parsing. T8 explicitly tests the garbage case → returns `winner=None` rather than crashing, but real-world drift will cost us judgments. Mitigation later: switch to JSON output via the SDK's `response_format` knob if available, or use the `tool_use` schema validator. Don't switch in this phase — adds complexity without proven need.
- **Judge non-determinism.** Same input on different runs can disagree. Set `temperature=0.0` (already in T1's `complete()` default). For multi-judge ensembles, see deferred Out-of-Scope.
- **Token cost.** 500 pairs × ~600 tokens (prompt + 2 responses + verdict) ≈ 300k tokens/cycle. At Sonnet 4.5 rates that's ~$1/cycle. Run cadence is bounded by Claude Code in P15.3.9 — if it fires too often we'll see it in cost dashboards. Not a blocker; flagged so the operator knows to watch.
- **Trace → judge coupling.** The judge needs **two responses for the same prompt** to compare. If only one trace per prompt exists in this repo's data, pairwise comparison can't happen for that prompt. Pointwise (`rate`) is the fallback, but Ψ-from-pointwise is noisier. P15.3.7 will need to handle the mix; flag for that PLAN.
- **`harness/brain/transport.py` drift.** Two separate transport implementations (extracted vs `harness/introspection/agent.py`) can drift over time. Acceptable for now — refactoring introspection is out of scope. If they meaningfully diverge, refactor introspection in a follow-up.
- **CLI subprocess robustness.** `claude --print` blocks until the response. If the subprocess hangs (network, large output), the judge stalls. Add a `subprocess.run(..., timeout=120)` in `_complete_via_cli`. Document the timeout in the helper's docstring.
- **Hard prereq P7.7.** Roadmap labels P7.7 as a hard prereq. In practice, the existing `harness/introspection/agent.py` already provides a working brain via `claude --print`. P7.7 is an *upgrade* (Agent SDK proper). This phase ships against the current transport; P7.7 lands later and only `harness/brain/transport.py` is touched. Roadmap language softened from "hard prereq" to "depends on a working transport which already exists".
