# RRT-ECHO Hypergraph Visualization

A read-only, interactive viewer for the artifacts produced by
[RRT-ECHO](../README.md). It renders the temporal hypergraph built from a long video —
events, participants, and the role bindings between them, unfolding over time — and lets a
`rrt.reader` QA run be traced back onto the graph, question by question.

The viewer **does not modify the pipeline and does not post-process the artifacts**: no
entity merging, no predicate rewriting, no number completion, no translation. Every event,
entity, and role shown on the page is exactly what is stored in `hypergraph.json`.

## Why this matters: binding errors, made visible

The paper frames long-video hallucination as **multimodal binding error** — every failure is
either *binding to nothing* (out-video: asserting content that never appeared) or *binding to
the wrong thing* (in-video: mis-associating content that did appear). The in-video failures
split into seven atomic modes:

| Mode | Binding error | What the viewer shows |
|---|---|---|
| R1 Predicate–Event Leakage | predicate ↔ event | an event's predicate vs. its actual time/participants |
| R2 Participant–Event Leakage | entity ↔ event | which entities an event's roles resolve to |
| R3 Role Permutation | entity ↔ role | the directed role written on each arrow |
| R4 Entity Continuity Failure | observation ↔ identity | which local instances are linked into one entity (the purple dashed circle) |
| R5 State–Version Leakage | entity ↔ state ↔ time | the state/attribute chips on a node, ordered by observed time |
| R6 Attribute–Owner Leakage | attribute ↔ entity | which owner each attribute/value fact hangs on |
| R7 Event–Relation Corruption | event_i ↔ event_j | the temporal order of events on the time bar |

The method suppresses these by construction: a fact only enters the hypergraph when its
participants, roles, time, and state are jointly observed and correctly bound, and answering
reads only the scoped subgraph for the query's `(entity, event, role, time, state)`. The
viewer is the debugging lens for *where* a binding went wrong — it separates, for every
question, the three failure locations:

- **not recorded** — the memory stored nothing in the gold span;
- **recorded but not retrieved** — the fact is in the graph but retrieval never surfaced it;
- **retrieved but answered wrongly** — the material was there and the reader still missed it.

## View your own run

```bash
python -m pip install pillow

# 1. build page data from a finished run
python visualization/hypergraph_viewer/build_data.py <run dir> "<display name>" \
    [--path-map /old/prefix/=/new/prefix/]

# 2. (optional) attach a reader QA run so answers trace back to the graph
python visualization/hypergraph_viewer/build_qa.py <qa dir> "<display name>" <questions jsonl> [annotation globs ...]

# 3. serve and open in a browser
python visualization/hypergraph_viewer/serve.py 8109
```

`<display name>` must match between the two build steps. `--path-map` rewrites the absolute
paths stored in the run (the run records the machine that produced it); it follows the same
convention as `tools/render_text_evidence.py` and may be repeated. See
[`hypergraph_viewer/README.md`](hypergraph_viewer/README.md) for the full interaction model
and caveats.

## QA trace

After step 2, a "QA trace" switch appears. It lists each question — the reader's pick, the
gold answer, how many facts were retrieved, and the in-graph audit verdict — and, when a
question is selected, leaves only the facts the reader was shown on the graph:

- magenta ticks = the times of those facts; black ticks / thick borders = facts the audit cited;
- each fact is tagged by which retrieval hypothesis pulled it in (the stem, or only an option);
- if the question file carries provenance, the gold-answer evidence span and each distractor's
  own time appear as chips; opening one shows **what the memory actually recorded in that
  span** and how many of those facts were retrieved.

This is the diagnostic view: it directly separates "not recorded", "recorded but not
retrieved", and "retrieved but answered wrongly".

## Not in this repository

`runs/` is an experiment artifact: `data.json` embeds the avatars cropped from source frames,
and `segment.webm` is a transcoded preview (~230 MB for a 25-minute run). Both are excluded by
`.gitignore`; re-run `build_data.py` on another machine to regenerate them.

## A note on the vLLM compatibility shim

`visualization/vllm_compat_proxy.py` is unrelated to the viewer. It is the shim we used to run
the pipeline against vLLM 0.19.1, which rejects the `structured_outputs:
{"disable_any_whitespace": true}` field that `correspondence.py` sends. The proxy only drops
that key when it carries no structured-outputs constraint, forwarding everything else
unchanged. Newer vLLM releases that accept the field do not need it.
