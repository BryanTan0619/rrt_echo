# Findings from inspecting one full run with the viewer

Subject: a 25:41 MrBeast Squid Game remake (`0e3GPea1Tyg`), one complete
`rrt-echo-offline` run plus one `rrt.reader --diagnostic` over 14 hand-written multiple-choice
questions (cross-time identity / role binding). Model: Qwen3-VL-8B-Instruct throughout.

These are **observations from one run**, not judgments on the method. Most of them change with
a larger model in place of the 8B.

## Run summary

| | |
|---|---|
| windows | 194 planned, 184 processed (15 rejected by structural validation: illegal boxes, undeclared regions, truncated identity comparisons) |
| instances → entities | 605 → 475, of which 82 link 2–6 local instances |
| identity decisions | 324 (141 same / 183 different); 1843 candidate pairs, 797 compared (budget exhausted) |
| facts | 649 event + 285 text; `relations` empty |
| answers | 5 / 14 correct; in-graph audit insufficient on all 14 |

## 1. The audit cannot cite instance descriptions — the most consequential one

**Observation.** q07 asks which vest number the first person to fall off the glass bridge
picked; the answer is 01. The instance at 17:08 is described as
`A woman with blonde hair wearing a green tracksuit with number 01 on her chest.` — it is
exactly the `agent` of `falls through the platform` at 17:10. The answer is closed in a single
record, and the reader got it right.

But the audit's legal citation IDs are only the 23 fact / identity_decision / relation IDs;
`audit_assertions.received` does not include instance IDs. **The one thing that proves the
answer has no legal ID to cite**, so the model can only cite the event fact without the number,
and the whole question is judged insufficient.

**Scale.** 319 of the 605 local instances have a number in their description; **210 of them
(66%) have no text/attribute/state fact attached** — the number lives only in the description.
The reader can see it and answer from it; the audit can cite none of it.

## 2. Co-observation hyperedges are also unciteable

q01's fourth assertion is `supported` with evidence chain
`["F9", "F10", "coobserved:b088dbe59922f11ce5a3"]` — two facts plus their co-observation
hyperedge. The `event_hyperedges` id is not in `received`, so one illegal ID voids the whole
chain, which is downgraded to insufficient with the chain cleared.

That assertion ("the elimination event and the 456 ELIMINATED caption are connected by
co-observation") is exactly what this structure is meant to express.

## 3. Without audio, speaker_identity assertions are necessarily insufficient

5 of the 31 assertions were `supported` by the model but downgraded by the code. Besides the
one above, the other 4 (q05 q11 q12 q14) were classified by the model as `speaker_identity`,
which requires citing a bound speech hyperedge. No audio was enabled in this run, so no such
hyperedge exists — **any assertion in that category is necessarily insufficient, independent
of the graph content**. The model is also unstable across the three claim types: in q01 and
q03 it labeled "someone did something" as `utterance_content`.

## 4. Retrieval is purely lexical; a gold answer with no literal match is systematically eliminated

`payload_for` turns the stem and each option into retrieval hypotheses, each with a
`top_k/(n+1)` quota. Recomputing q09 ("what shape did 067 carve in the honeycomb round", gold
Circle) with `retrieve()`:

| hypothesis | extra facts beyond the stem base |
|---|---|
| stem | 10 base facts, all unrelated (the 406 elimination at ~23:00; only the word `player` matched) |
| option A Umbrella | **5**, incl. 4:54 `I KNOW UMBRELLA GOIN HOME`, 6:44 `YOU DID THE UMBRELLA!?` |
| option B Star | 0 |
| **option C Circle (gold)** | **0** |
| option D Triangle | 1 (5:13 someone carving a triangle into the disc with a needle) |

`circle`, `shape`, `carve`, `honeycomb`, `MrBeast` hit 0 of the 934 facts. The only material
about someone successfully making a shape is the two umbrella captions, owned by the host,
without a number. **On this material, option A is the only defensible choice — any model
would pick it.** `terms()` merges only a dozen verb forms and does not normalize noun
synonyms, so whether an answer is retrievable depends on whether the question writer's word
and the VLM's word happen to collide.

## 5. With complete material, the reader still misses temporal ordering

q08 asks who was eliminated first in round 1 of musical chairs, gold 075. The material has four
elimination records with timestamps in the right order: 22:53 `075 ELIMINATED` → 23:14
`406 ELIMINATED` → 23:34 and 23:42 `456 ELIMINATED`. The reader picked 456 (the only record
that repeats). This question needs no graph structure at all — only comparing numbers.

## 6. Number readings are unreliable and actively create wrong bindings

- q06's gold is 067 being invited by the host to chant. The instance at 1:47 is described as
  `number 06`; `06` appears 10 times across the film and `067` 3 times (all in the marbles
  round). `06` could be a truncation of 060–069 or 106/206/306/406 — **nothing can complete it
  to 067**.
- In q09, the celebrating person at 6:39 is recorded as `035`, matching neither the 067 asked
  about nor the 381 annotated as gold.

Errors of this kind are worse than omissions: an omission only fails to answer, a misread
reassigns the event to the wrong person downstream.

## 7. The same person is judged to be different entities (the flip side of q07)

Between 16:15 and 17:20 the same blonde woman appears in four local instances (16:24 picking a
vest, 16:57, 17:03, 17:08 falling). **All four are `local_only`, mutually unlinked.** 17:03 and
17:08 are one second apart yet are different entities.

Checking the identity decisions: these instances participated in 6 pairwise comparisons, **all
judged `explicit_distinctness`** — not missed, but compared and judged different. With hundreds
of people in identical green tracksuits, the 8B has no distinguishing information from
appearance alone, and the rule orientation prefers "different" over a wrong merge. That
orientation is sound; its cost is that cross-time two-hop questions cannot be answered.

## Reading the 5/14 by failure location

Classifying by where the failure happened is more useful than the total score:

| where it failed | questions |
|---|---|
| not recorded; gold answer literally zero-hit | q05 q09 q11 |
| recorded but unciteable / unlinkable | q07 |
| material complete; reader answered wrongly | q08 |
| hit by luck, no real evidence | q02 q03 |

Re-running the reader over the same `hypergraph.json` with a stronger reader would separate
"memory problem" from "reader problem" without touching the memory itself — a control we have
not run yet.

## Reproduce

```bash
# build (parameters for a full pass; the README example values are for a short clip)
python -m rrt_echo.rrt.pipeline --video <mp4> --source-video-id <id> --out <run> \
  --profile simple --mode images --model Qwen3-VL-8B-Instruct --base-url <vllm>/v1 \
  --sample-fps 2 --local-seconds 12 --processor-max-pixels 401408 \
  --max-local-calls 400 --max-identity-batches 200 --association-workers 2 \
  --max-tokens 4096 --timeout 120 --budget 10800

# answer
python tools/inspect_memory.py --graph <run>/hypergraph.json --question '...' --out <inspect>
python -m rrt_echo.rrt.reader --graph <run>/hypergraph.json --questions <questions.jsonl> \
  --acceptance <inspect>/acceptance.json --diagnostic --out <qa> \
  --model Qwen3-VL-8B-Instruct --base-url <vllm>/v1 --workers 2
```

Two vLLM notes: the multi-image cap must be ≥ the frames per window (we set 64; the default 32
errors with "At most 32 image(s)"), and the `structured_outputs` field compatibility issue is
described in `vllm_compat_proxy.py`.
