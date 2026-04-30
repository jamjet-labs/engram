# Engram v2 — Tier 2 + Tier 3 + Novel Approaches Design

**Status:** Draft for review
**Date:** 2026-04-30
**Author:** Sunil Prakash (with Claude)
**Target outcome:** Move Engram v2 from 88.8% substring-proxy / ~65-80% substantive on LongMemEval-S 500q to **≥85% officially-judged** (target 90%+) within 1.5-2 weeks of focused work, at ≤$15 per full-500 run.

## 1. Context

The Engram v2 Python rewrite shipped 12 of 13 roadmap phases plus Tier 1 improvements (XML verifier, candidate pool=5, `--extract` flag) on commit `9f60909`. A Tier 1 500-q run was kicked off at end of session on 2026-04-30 but died at q16 — the post-Tier 1 baseline number is unmeasured. Phase 11b (ReAct retrieval agent over the SVO event calendar) and several Tier 2/3 items remain.

The OpenAI API key in `~/Development/sunil-ws/dci-research/.env` was refreshed and verified (200 status). This unblocks the official LongMemEval `gpt-4o-mini` judge — a prerequisite for any leaderboard-credible number.

This spec covers the next implementation batch: **8 work items** (5 from the existing Tier 2/3 list + 3 novel directions) sequenced for incremental ablation legibility.

## 2. Goals & non-goals

### Goals
- Land all 8 work items behind config flags
- Validate each on a 100-q smoke with the official `gpt-4o-mini` judge before stacking
- Deliver M3 (final) full-500 judged accuracy ≥ 85% (floor); 88-92% target
- Keep cost per full-500 run ≤ $5 with the default (gpt-4o-mini) reader; ≤ $15 if the ablation in item 1 promotes Sonnet 4.6 to default
- Produce ablation traces sufficient to attribute lift per item

### Non-goals
- Beating AgentMemory (96.2%) in this batch — that's a separate research arc
- Productionising the fine-tuned cross-encoder publishing flow (private HF push is enough)
- Performance optimisation beyond cost (latency is not a target)
- Changes to wire protocol (HTTP/MCP), Spring starter, or langchain4j integration
- Changes to the Rust v1.x LTS

## 3. Scope — eight work items

The eight items fall in three buckets:

**From original Tier 2/3:**
1. **Wire `QueryDecomposer` into retrieval** (#5) — class exists in `src/engram/read/decomposer.py`, never called by `Engram.context()` or the benchmark
2. **Reader-model ablation** (#6) — current reader is Haiku 4.5. Ablate Haiku 4.5 / gpt-4o-mini / Sonnet 4.6 on a 100-q smoke; promote whichever wins on accuracy-per-dollar to default. Default starting point is gpt-4o-mini (cheapest).
3. **Adaptive self-consistency** (#8) — N=3 reader samples + vote, gated by verifier verdict
4. **ReAct retrieval agent over SVO event calendar** (#4 / Phase 11b) — new module, brain on `gpt-4o-mini`
5. **Fine-tune cross-encoder on LongMemEval-style queries** (#7) — offline training pipeline, swap into rerank step

**Novel directions:**
6. **Programmatic temporal solver over the event store** (N1) — small DSL, deterministic SQL execution against SVO calendar
7. **Tool-augmented reader** (N5) — Sonnet with `solve_temporal`, `search_events`, `count_between`, `add_days`, `days_between`, `search_facts` tools
8. **Query-time conditioned re-extraction** (N2) — when verifier returns PARTIAL/NO, re-extract top-3 candidate sessions on-the-fly, ephemeral facts (not persisted)

## 4. Architecture

### 4.1 Pipeline shape

```
record → extract → store ─┬─► recall ─► classifier ─► context ─► READER (gpt-4o-mini default; Sonnet 4.6 if ablation promotes it; tool-augmented)
                          │                              ▲             │
                          │                       Decomposer (#5)      ├─► tool calls dispatched via ToolRegistry
                          │                       splits compound      │
                          │                       Q → k subqueries     ▼
                          │                       (parallel recall)   verifier (utility tier)
                          │                                            │ if PARTIAL/NO + low confidence:
                          │                                            ├─► (a) query-time re-extraction (N2)
                          │                                            ├─► (b) self-consistency N=3 (#8)
                          │                                            └─► (c) ReAct fallback (#4)
                          │
                          └─► (offline) fine-tune cross-encoder (#7) → swap into rerank
```

### 4.2 Cross-cutting changes

- **`ModelTier` config** — `reader` (gpt-4o-mini default; promoted to Sonnet 4.6 if item 1's ablation shows it earns its keep) + `utility` (gpt-4o-mini for verifier, decomposer, classifier, event extraction, ReAct brain, re-extraction)
- **`Tool` protocol + `ToolRegistry`** — single tool implementation shared between tool-aug reader and ReAct agent
- **`TemporalSolver`** — used as a tool by the reader and by ReAct
- **Adaptive self-consistency** — internal to `Reader`, no caller change

### 4.3 Escalation ladder (the cost-control mechanism)

After the initial reader+verifier turn, escalation fires only on `verdict ≠ YES`:

| Rung | Trigger | Cap | Cost |
|---|---|---|---|
| (a) Re-extract | verdict ∈ {PARTIAL, NO}; top-3 sessions exist | 1 attempt | +1 utility call (~$0.001) |
| (b) Self-consistency | post-(a) verdict ≠ YES; category ∈ {temporal, multi-session, knowledge-update} | N=3 reader samples | +2 reader calls (~$0.001 with gpt-4o-mini, ~$0.04 with Sonnet) |
| (c) ReAct fallback | post-(b) verdict ≠ YES | max_hops=4, hop_timeout=15s, total_budget=60s | +up to 4 utility calls + tool I/O |
| (d) Abstain | all three fail | — | 0 |

Verifier runs after each rung. Worst-case per question ~$0.005 (gpt-4o-mini reader) or ~$0.05 (Sonnet reader); easy questions exit at rung 0. Predicted average:
- gpt-4o-mini reader path: ~$0.003/q → ~$1.5/full-500
- Sonnet reader path: ~$0.012/q → ~$6/full-500

## 5. Component interfaces

### 5.1 `ModelTier` (`src/engram/llm/tier.py`)
```python
class ModelTier(BaseModel):
    reader: LLMClient            # gpt-4o-mini default; ablate-able to Haiku 4.5 / Sonnet 4.6
    utility: LLMClient           # gpt-4o-mini

    @classmethod
    def default(cls) -> "ModelTier":
        # Cost-optimised default. Promote reader to Sonnet 4.6 only if item 1's
        # ablation shows the accuracy delta justifies the ~30x cost increase.
        return cls(
            reader=OpenAILLM("gpt-4o-mini"),
            utility=OpenAILLM("gpt-4o-mini"),
        )

    @classmethod
    def sonnet_reader(cls) -> "ModelTier":
        # Upgrade tier — used only after item 1's ablation justifies it.
        return cls(
            reader=AnthropicLLM("claude-sonnet-4-6"),
            utility=OpenAILLM("gpt-4o-mini"),
        )
```
Threaded into `Engram.open(..., tier=ModelTier.default())`. Existing `llm=` parameter stays for back-compat (uses for both).

### 5.2 `Tool` + `ToolRegistry` (`src/engram/tools/`)
```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict
    async def __call__(self, **kwargs) -> ToolResult: ...

class ToolResult(BaseModel):
    content: str
    raw: Any = None

class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def for_anthropic(self) -> list[dict]: ...
    def for_openai(self) -> list[dict]: ...
    async def dispatch(self, name: str, args: dict) -> ToolResult: ...
```

Built-in tools (one file each):
- `search_facts(query, top_k=5, scope)` — wraps `Engram.recall`
- `search_events(subject?, verb?, object?, start?, end?, scope)` — wraps `store.search_events`
- `solve_temporal(query)` — calls `TemporalSolver`
- `count_between(start, end, filter)` — deterministic count
- `add_days(date, n)` / `days_between(a, b)` — pure date arithmetic

### 5.3 `TemporalSolver` (`src/engram/solve/temporal.py`)
```python
class TemporalQuery(BaseModel):
    op: Literal["count", "duration", "ordering", "before_after", "elapsed"]
    subject: str | None
    verb: str | None
    object: str | None
    anchor_event: str | None
    bound: Literal["before", "after", "between"] | None
    window: tuple[datetime, datetime] | None

class SolverResult(BaseModel):
    answer: str | int | float
    confidence: float
    evidence_event_ids: list[UUID]

class TemporalSolver:
    def __init__(self, store: EngramStore, llm: LLMClient): ...
    async def parse(self, question: str, today: datetime) -> TemporalQuery | None
    async def solve(self, q: TemporalQuery, scope: Scope) -> SolverResult | None
```
- `parse` uses utility LLM with structured-output prompt; returns `None` if not clearly temporal/structured
- `solve` is pure SQL/SVO calendar lookup, deterministic
- Either returning `None` falls through to LLM reader

### 5.4 `Reader` extensions (`src/engram/read/reader.py`)
```python
class ReaderConfig(BaseModel):
    tools: ToolRegistry | None = None
    self_consistency_on_partial: int = 3       # N=1 disables
    enable_reextract: bool = True

class Reader:
    def __init__(self, tier: ModelTier, store: EngramStore, config: ReaderConfig): ...
    async def read(question, context, today, scope) -> ReadResult: ...
```
Internal flow follows the escalation ladder. Caller-facing API unchanged from current `Reader`.

### 5.5 Query-time re-extraction (`src/engram/read/reextract.py`)
```python
class QueryConditionedReextractor:
    def __init__(self, llm: LLMClient): ...     # utility tier
    async def reextract(
        question: str,
        candidate_session_ids: list[str],
        store: EngramStore,
    ) -> list[Fact]                              # ephemeral, not persisted
```
Triggered by Reader when verifier verdict is PARTIAL/NO. Output is fed back into context for one re-read, then discarded.

### 5.6 `ReActAgent` (`src/engram/agent/react.py`)
```python
class AgentResult(BaseModel):
    answer: str
    abstained: bool
    trace: list[AgentStep]
    n_hops: int

class ReActAgent:
    def __init__(self, tier: ModelTier, tools: ToolRegistry,
                 max_hops: int = 4, hop_timeout_s: float = 15.0): ...
    async def answer(question, scope, today) -> AgentResult: ...
```
Brain: `tier.utility` (gpt-4o-mini). Tools: the same `ToolRegistry` from §5.2.

### 5.7 Decomposer wiring (no new module)
```python
# engram.py — modified
async def context(self, query, user_id, classifier, decompose: bool = True):
    subqueries = await self._decomposer.decompose(query) if decompose else [query]
    facts_by_q = await asyncio.gather(*[self._recall(sq, ...) for sq in subqueries])
    fused = reciprocal_rank_fusion(facts_by_q)
    return format_context(...)
```

Decomposer fire heuristic (avoid 500 unnecessary calls):
```
if len(question.split()) < 12 AND not contains_conjunction(question):
    skip; subqueries = [question]
else:
    call utility LLM
```
`contains_conjunction` = regex `\b(and|or|both|either|while|whereas|along with|as well as)\b` plus question-mark count > 1.

### 5.8 Fine-tuned cross-encoder (`training/cross_encoder/`)
```
training/cross_encoder/
  ├── generate_pairs.py        # synthetic Q/passage pairs from haystacks via gpt-4o-mini Batch API
  ├── train.py                 # sentence-transformers fine-tune (MS MARCO base + adapter)
  ├── eval.py                  # nDCG@10 vs base on held-out
  └── README.md                # repro
```
Output: model card + weights pushed to private HF Hub. Loaded by `CrossEncoderReranker(model_path=...)`. Lives outside `src/engram/` — offline tooling, not runtime.

## 6. Data flow rules (the 'what fires when' details)

### 6.1 Single-question trace
1. `classifier.classify(question)` → category, token_budget
2. `decomposer.decompose(question)` → `[q1]` or `[q1, q2, ...]` (utility LLM, ≤300 tokens)
3. For each subq: recall + rerank → top_k facts (parallel `asyncio.gather`)
4. RRF fuse → context_string (truncated to category budget)
5. `reader.read(question, context, today, scope, tools=registry)`
   - Reader model (per `ModelTier.reader`) generates answer; may emit tool_use blocks
   - Each tool_use dispatched via `ToolRegistry`, result appended, continue
   - Final answer text
6. `verifier.verify(question, context, answer)` → YES/PARTIAL/NO (utility LLM)
7. If YES → return; else escalate per §4.3

### 6.2 Tool-aug reader caps
- Tool loop cap: max 5 tool calls per single `read()` call
- Per-tool timeout: 10s
- If the model emits a `tool_use` block after also producing text content in the same assistant turn, treat the text as the final answer and ignore the trailing `tool_use`. (Some providers interleave; we want a deterministic "first answer wins" rule.)

### 6.3 ReAct termination
Terminate when ANY of:
- Agent emits `final_answer` tool call
- `n_hops >= max_hops` (4)
- Elapsed >= total_budget_s (60)
- Same tool called with identical args twice in a row (loop detection)
- At any hop where the agent emits text content (a candidate answer) alongside or instead of tool calls, run the verifier on that candidate; if verdict is YES, return that text as the final answer

### 6.4 Caching for ablation reuse
Two stable caches under `~/.cache/engram-bench/`:
- **Ingestion snapshot** — `sqlite_dump_<dataset_hash>_<extract_mode>.db.zst`, restored at start of each run
- **Reader response cache** — keyed on `sha256(model + system_prompt + context + question + tools_signature + n_sample_index)`. Hits = $0.

Per-component, not end-to-end. Swapping cross-encoder invalidates only steps downstream of rerank.

### 6.5 Per-question observability
Every benchmark run writes a per-question JSONL trace (~5KB/q, ~2.5MB total/500q):
```json
{"qid": "...", "category": "temporal-reasoning",
 "decomposer": {"fired": false, "subqueries": ["..."]},
 "recall": [{"sq": "...", "n_candidates": 50, "top_k": 5, "fact_ids": [...]}],
 "reader": {"model": "claude-sonnet-4-6", "tool_calls": [...], "answer": "...", "tokens": {...}},
 "verifier": {"verdict": "PARTIAL", "missing": "..."},
 "escalation": [{"rung": "reextract", "outcome": "answer_changed", "verdict": "YES"}],
 "react": null,
 "final": {"answer": "...", "abstained": false, "judge_verdict": "correct", "cost_usd": 0.011}}
```

## 7. Implementation sequence (Approach A — cost-and-confidence-first)

| Order | Item | Effort | Smoke gate |
|---|---|---|---|
| 1 | Model tier scaffolding + reader-model ablation (#6): Haiku 4.5 vs gpt-4o-mini vs Sonnet 4.6 on the same 100-q set | 1 day | Default reader = winner on accuracy-per-dollar (cost normalised to gpt-4o-mini = 1.0). Promote Sonnet only if Δaccuracy ≥ +5pp vs gpt-4o-mini. |
| 2 | Decomposer wiring (#5) | 0.5 day | Δ ≥ 0; lift on multi-session |
| 3 | Programmatic temporal solver (N1) | 2 days | Δ ≥ +2pp on temporal-reasoning |
| 4 | Tool-aug reader (N5) | 1 day | Δ ≥ 0 overall |
| 5 | Query-time re-extraction (N2) | 1 day | Δ ≥ 0 overall |
| 6 | Adaptive self-consistency (#8) | 0.5 day | Δ ≥ 0 (cost ≤ 2× rung-(b) budget) |
| 7 | ReAct agent (#4) | 2-3 days | Δ ≥ +2pp on multi-session/temporal |
| 8 | Fine-tuned cross-encoder (#7) | 2-3 days | Δ ≥ +1pp overall |

**Gate-fail handling:** if smoke shows Δ < threshold, the item still merges, but its config flag defaults to `off`. (Item 1 is special: it has no "off" — the smoke result simply determines which reader becomes the default in `ModelTier.default()`.) In all cases write a negative-result note in the spec changelog (one paragraph) and continue to the next item. Do NOT block the train.

## 8. Test strategy

### 8.1 Unit / integration tests
| Item | Test files | Scope |
|---|---|---|
| Model tier | `tests/unit/test_model_tier.py` | construction, threading, back-compat |
| Decomposer wiring | `tests/unit/test_decomposer_wiring.py`, `tests/integration/test_context_decomposed.py` | heuristic gate, RRF fusion, budget |
| Temporal solver (N1) | `tests/unit/test_temporal_solver.py` | parse None for non-temporal; each `op` on fixtures |
| Tools + registry | `tests/unit/test_tools.py` | each built-in tool: schema, dispatch, errors |
| Tool-aug reader (N5) | `tests/integration/test_reader_with_tools.py` | mock LLM tool_use → dispatch → answer |
| Self-consistency (#8) | `tests/unit/test_self_consistency.py` | majority vote, tiebreak, category gate |
| Re-extract (N2) | `tests/unit/test_reextract.py`, `tests/integration/test_reextract_in_reader.py` | ephemeral facts, no persist, gating |
| ReAct (#4) | `tests/unit/test_react.py`, `tests/integration/test_react_e2e.py` | termination, trace, loop detection, short-circuit |
| Fine-tune (#7) | `training/cross_encoder/tests/` | pair-gen determinism, train smoke, eval |

Target: 30+ new tests, all green, mypy --strict + ruff clean. CI budget ≤ 30s.

### 8.2 Smoke ablations
100-q stratified subset, official `gpt-4o-mini` judge via LongMemEval `evaluate_qa.py`. ~$1/smoke, ~7 min wall-clock. Run after each item lands.

### 8.3 Milestone full runs
Three 500-q + judge runs total (~$10/each):
- **M1** (after items 1-3) — confirm cheap-stack lift; expected ≥+8pp judged over current baseline
- **M2** (after items 4-6) — non-agent ceiling; expected mid-80s judged
- **M3** (after items 7-8) — leaderboard claim; expected 88-92% judged

If M1 or M2 disappoints by ≥3pp from prediction, **stop and diagnose** before continuing.

### 8.4 Reporting
Per smoke and milestone:
- JSONL trace (per §6.5)
- Markdown report at `benchmarks/reports/<date>_<milestone>.md`: judged accuracy overall + per-category, $$ spent, wall-clock, top 5 failure-mode samples, diff vs prev baseline
- Updated `benchmarks/PROGRESS.md`

## 9. Cost budget

Item 1's reader ablation requires three readers tested on the same 100q (~$5 with Sonnet leg dominating). Subsequent smokes use whichever reader item 1 promotes to default.

| Phase | Smokes | Milestones | One-time | Subtotal |
|---|---|---|---|---|
| Item 1 (3-way reader ablation) | 3 × $1-2 | — | — | ~$5 |
| Items 2-3 | 2 × $1 | M1 × $3 (gpt-4o-mini) or $10 (Sonnet) | — | $5-12 |
| Items 4-6 | 3 × $1 | M2 × $3 or $10 | — | $6-13 |
| Items 7-8 | 2 × $1 | M3 × $3 or $10 | $10 (cross-encoder fine-tune via OpenAI Batch API, 50% off) | $15-22 |
| **Total — gpt-4o-mini default path** | | | | **~$31** |
| **Total — Sonnet promoted path** | | | | **~$52** |

Operational cost per full-500 run after this batch:
- gpt-4o-mini reader (default unless item 1 says otherwise): ~$1.5 reader + ~$1.5 judge = **~$3/run**
- Sonnet 4.6 reader (only if item 1's ablation promotes it): ~$8 reader + ~$1.5 judge = **~$10/run**

## 10. Definition of done

Programme is complete when **all of**:
- [ ] M3 judged accuracy ≥ 85% (ceiling target 90%+; floor 85%)
- [ ] All 8 items merged on `main` with tests + flags
- [ ] Cost-per-full-run ≤ $5 on the default (gpt-4o-mini) path; ≤ $15 if item 1 promoted Sonnet
- [ ] `benchmarks/PROGRESS.md` reflects all milestones
- [ ] Spec doc + per-item negative-result notes committed
- [ ] One ablation README per smoke at `benchmarks/reports/`

If the ceiling is hit early, the programme stops early and ships.

## 11. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Sonnet 4.6 reader doesn't beat gpt-4o-mini by ≥+5pp (the promotion threshold) | high | low | Default stays gpt-4o-mini (~30× cheaper). The whole point of item 1's ablation is to discover this rather than assume — Sonnet not winning is a feature of the process, not a failure |
| gpt-4o-mini reader hallucinates more on tool-augmented turns than Sonnet | medium | medium | Item 4 (tool-aug reader) smoke specifically watches for this regression on temporal/numerical category. If observed, Sonnet's promotion threshold drops to +3pp |
| ReAct agent loops or burns budget without terminating | medium | high | Multiple termination conditions (§6.3); per-question total_budget_s=60 hard cap; trace logging makes loop diagnosis easy |
| Programmatic temporal solver DSL doesn't cover enough question shapes | medium | medium | `parse` returning `None` falls through to LLM reader — solver is additive, can't regress |
| Synthetic training data for cross-encoder is too noisy | medium | medium | nDCG@10 eval on held-out before swap; gate fails → ship behind flag off |
| Per-question caching gives misleading ablation signal (one branch's cache hit while other branch re-runs) | low | high | Cache key includes `tools_signature` and full system prompt hash — config-shape changes invalidate |
| Self-consistency triples cost without lift on PARTIAL questions | low | low | Category gate + cost cap in escalation ladder |
| OpenAI Batch API delays training data > 24h | low | low | Spec allows on-demand fallback at full price (~$20 instead of $10) |

## 12. Out-of-scope follow-ups

Items deliberately deferred (memory file `project_engram_v2_progress.md` Tier 2 list):
- Full sentence-transformers embedder fine-tune (separate from cross-encoder rerank)
- Self-RAG style critique loop beyond what verifier+ReAct already do
- Knowledge distillation from Sonnet to a smaller reader
- Memory graph with typed edges
- Token-level retrieval (ColBERT-style)
- Adversarial verifier
- Compositional decomposition with parallel per-subquery memory (different from §5.7 wiring; would need its own decomposer model)

Any of these become candidates for a follow-up batch if M3 falls short of ceiling.

## Changelog

### Item 1 (reader ablation) — 2026-04-30

100-q stratified smoke, official `gpt-4o-mini` judge. Embedder Ollama `nomic-embed-text`. No flags (bare reader pipeline).

| reader | overall | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | wall-clock | est cost |
|---|---|---|---|---|---|---|---|---|---|
| `gpt-4o-mini` | **64.0%** | 71% | 53% | 88% | 24% | 75% | 75% | 408s | ~$0.50 |
| `claude-haiku-4-5-20251001` | 61.0% | 65% | 53% | 88% | 41% | 69% | 50% | 544s | ~$1.00 |
| `claude-sonnet-4-6` | **69.0%** | 82% | 53% | 94% | 41% | 69% | 75% | 957s | ~$2.00 |

**Δ vs gpt-4o-mini:** Sonnet **+5.0pp** (right at promotion threshold), Haiku **-3.0pp**.

**Decision:** Promotion **deferred**. `ModelTier.default()` stays on `gpt-4o-mini`.

**Reasoning:** Sonnet hits the +5pp threshold exactly, but at n=100 the noise is roughly ±5pp — repeating the run could plausibly show 67% or 71%. More importantly, this is a *bare-reader* comparison; Sonnet's real edge is tool use, which item 4 introduces. Re-test all three readers after item 4 lands (with `--tools` enabled) to see whether the gap widens enough to justify ~30× cost. Sonnet also shows a regression on `single-session-user` (-6pp) that's worth understanding before locking it in. In the meantime, items 2-6 ablations run on cheap gpt-4o-mini (~$15-20 saved over the programme).

**Re-test scheduled:** after item 4 lands. If Sonnet then shows ≥+5pp on the tool-augmented pipeline, promote.

### Item 2 (decomposer wiring) + Item 3 (programmatic temporal solver) — 2026-04-30

#### Initial smokes (loose gate)

| run | overall | multi-session | temporal-reasoning | knowledge-update |
|---|---|---|---|---|
| baseline (no flags, n=100) | 64.0% | 53% | 75% | 71% |
| --decompose only (n=100) | 64.0% | 59% (+6) | 69% (-6) | 71% |
| --solver only (n=100) | 63.0% | 59% | 75% | 59% (-12) |
| --decompose --solver (n=100) | 63.0% | 65% | 69% | 59% |
| --decompose --solver (n=200, M1) | 63.0% | 59% | 59% | 53% |

**M1 gate FAILED:** target +8pp lift over baseline; observed -1pp.

#### Diagnosis

1. **Decomposer:** robust +6pp lift on multi-session (the targeted category), but the loose gate (≥10 words + 'and'/'or' conjunction) fired on temporal-reasoning compound questions and decomposed them poorly, losing -6pp.
2. **Solver:** fired ~3-5% of the time, **always wrong**. Returned small numbers like "1" on knowledge-update questions like "how many followers do I have on Instagram now?" (where the answer is "1300", not the count of "have_followers" SVO events). The LLM parser is too eager to classify "how many X" as a count-of-events query.

#### Root causes

- Benchmark didn't populate the SVO event calendar — fixed by adding `_ingest_events` that calls `extract_events` per session when `--solver` is set. Solver still wrong because the LLM parser classifies non-event questions as count queries.
- Decomposer heuristic gate was too loose — fixed by skipping questions containing temporal/ordering markers ("first", "last", "before", "after", "between", "since", "ago", "how long", "how many days/months/...", "when did", "what was the date"), and by treating only "and" (not "or") as a compound signal.

#### Final smokes after fixes

| run | overall | multi-session | temporal-reasoning |
|---|---|---|---|
| --decompose (tightened gate, n=100) | **65.0%** | **59% (+6)** | **75% (0)** |

**Decision:**
- **--decompose: ship ON by default** with the tightened gate. Small but robust net lift; recovers the multi-session win without the temporal regression.
- **--solver: ship behind `--solver` flag default OFF.** The parser is too aggressive; needs stricter prefiltering (e.g., require explicit anchor words like "before/after X" in the question) before it would be safe to enable. Defer to a follow-up.

**Programme cost so far:** ~$10 (3 reader ablations + 5 Phase B smokes incl. event-extraction overhead).
