# RRT–ECHO

Build a traceable event memory from a long video, then answer questions with binding-constrained
retrieval over that memory.

This is the **base implementation**, running the default `simple` pipeline. RRT extracts local
events and entity correspondences; ECHO stores joint facts, evidence, and revisable identity
interpretations. Answering and evidence support are scored separately; a successful build is not
video semantic acceptance.

The method is grounded in one idea: long-video hallucination is a **binding error** — either
binding to nothing (out-video: asserting content that never appeared) or binding to the wrong
thing (in-video: mis-associating content that did appear). The temporal hypergraph suppresses
both **by construction**: a fact only enters the graph when its participants, roles, time, and
state are jointly observed and correctly bound, and answering reads only the scoped subgraph for
the query. The in-video failures split into seven atomic modes: R1 predicate–event leakage,
R2 participant–event leakage, R3 role permutation, R4 entity continuity failure, R5 state–version
leakage, R6 attribute–owner leakage, and R7 event–relation corruption.

## 1. Method and code

```text
MP4 ──→ frame sampling & contiguous segments ──→ RRT joint observation ──→ budgeted entity correspondence
         ↑ optional ASR text                          ↓
                                          ECHO observation ledger & binding acceptance
                                                              ↓
                                                      hypergraph.json
                                                              ↓
                              scoped retrieval → QA → independent in-graph support audit
```

| Module | Responsibility | Main code (`src/rrt_echo/`) |
|---|---|---|
| RRT | Per-segment joint extraction of people/objects/regions, event roles, text, time, and source frames; compares `continues`, `same_identity`, `part_of` | `rrt/pipeline.py`, `rrt/observation.py`, `rrt/association.py` |
| ECHO | Appends raw observations; validates ownership and identity conflicts; produces the hypergraph and entity registry | `rrt/memory.py`, `rrt/hypergraph.py`, `echo_perception/graph.py` |
| Retrieval | Retrieves whole facts by question and time, completing identity, ownership, and evidence dependencies | `retrieval.py`, `rrt/binding_scope.py`, `rrt/reader.py:payload_for` |
| QA | Reads only the graph, never the video; forces an answer, then audits its support separately | `rrt/reader.py`, `evaluation.py` |
| Optional audio | Timestamped ASR and anonymous voice clusters; unresolved speakers stay unresolved | `rrt/audio.py`, `rrt/multimodal.py` |

`schema.py`, `memory.py`, `identity.py`, `storage.py` are the shared graph-storage base;
`echo_perception/` and `perception/` keep the sampling, validation, and model adapters the main
pipeline depends on. Some compatibility implementations remain internally — use the entry points
below. Historical experiments and old reports are not shipped with this repository.

### What the hypergraph stores

Local facts keep their role endpoints separate from the global identity interpretation. A writing
event stores `writer=p1, instrument=pen1, surface=arm1`; a text observation stores `owner=arm1,
value="18 km"` with `part_of(arm1,p1)` explaining the body ownership. Cross-segment
`same_identity(p1,p7)` explains them as one entity.

Roles, time, regions, and evidence of the same event are retrieved together; facts are never
stitched into a new event from text co-occurrence alone. Text and action sharing a carrier and
frames may form a co-observed attribute bundle, but this does not prove the text was written by
that action. Identity revisions never overwrite the original local endpoints.

## 2. Install

Python 3.11+, from the repository root:

```bash
git clone https://github.com/BryanTan0619/rrt_echo.git
cd rrt_echo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,perception]'
pytest -q
```

The graph storage and retrieval core has no third-party runtime dependency; the video pipeline
uses PyAV, Pillow, NumPy, and JSON Schema. The VLM service is deployed separately; this
repository ships no model weights, videos, or QA data.

## 3. Build a graph from an MP4

Prepare an OpenAI-compatible VLM service with image input and JSON output. The example uses image
mode explicitly; the server does not need this project's video transport extension installed.

```bash
export RRT_VLM_BASE_URL=http://127.0.0.1:8001/v1
export RRT_VLM_MODEL=your-vision-model
# if the service requires auth: export RRT_VLM_API_KEY=...

rrt-echo-offline \
  --video /path/to/video.mp4 \
  --source-video-id video01 \
  --out outputs/video01 \
  --profile simple --mode images \
  --model "$RRT_VLM_MODEL" --base-url "$RRT_VLM_BASE_URL" \
  --sample-fps 2 --local-seconds 12 \
  --processor-max-pixels 401408 \
  --max-local-calls 100 --max-identity-batches 24 \
  --association-workers 2 --max-tokens 4096 --timeout 120 --budget 3600
```

The output directory must not exist. `source-video-id` must match the `video_id` in the QA data.
Budget and call caps may cause partial coverage; check `report.json`, `coverage.json`,
`failures.json`, and do not treat `DONE.json` as a quality pass. The timeout budget is not a hard
kill guarantee for the remote model service.

The default `simple` profile does not enable the global story index, multi-round local repair, or
extra speaker audiovisual verification; it still performs local joint observation and budgeted
correspondence comparison. `--profile legacy` keeps part of the multi-stage experimental ability
and is not the base usage path.

```text
outputs/video01/
  ledger/observations.jsonl   # canonical: appended observations, bindings/revocations, audio, revisions
  hypergraph.json             # full snapshot used by QA, with snapshot_id
  entity_registry.json        # rebuildable entity registry; isolated instances stay unresolved
  storyline.json              # a time-ordered fact view, not a newly generated story
  facts.jsonl, bindings.jsonl # derived exports, not a second independent fact source
  view_manifest.json
  coverage.json, report.json, failures.json, request_inventory.json
  ...                         # source frames, request/response and diagnostic artifacts
```

## 4. Inspect retrieval, then run QA

First export the same scoped retrieval payload QA uses, without calling the model:

```bash
python tools/inspect_memory.py \
  --graph outputs/video01/hypergraph.json \
  --question 'Who is carrying the child near the end?' \
  --out outputs/video01_inspect
```

Retrieval uses term weighting, question time ranges, and full fact closure — no vector database.
A multiple-choice question's options can be retrieved as separate hypotheses but never become
facts. Results include event roles, identity chains, ownership chains, evidence IDs, and gaps;
unresolved endpoints stay unresolved.

Prepare a JSONL, one question per line:

```json
{"question_id":"q1","video_id":"video01","hallucination_type":"R2","question":"Who carries the child?","options":{"A":"The adult in the plaid shirt","B":"The woman in white"},"ans":"A"}
```

The example only shows the format; use your own video's questions and annotations. It supports an
`options` dict, `A) ...` options inline in the stem, and `question_type="True/False"`.
`hallucination_type` is used for grouped statistics and does not control the build; `ans` is used
only for scoring and is never sent to the model.

```bash
rrt-echo-qa \
  --graph outputs/video01/hypergraph.json \
  --questions /path/to/questions.jsonl \
  --acceptance outputs/video01_inspect/acceptance.json \
  --diagnostic --out outputs/video01_qa \
  --model "$RRT_VLM_MODEL" --base-url "$RRT_VLM_BASE_URL" \
  --workers 2
```

The acceptance produced by the inspect tool is explicitly `passed=false`, so `--diagnostic` is
used here. A real `passed=true` record can only be provided after video acceptance of the
corresponding snapshot; do not flip it just to run QA.

Each question normally costs two model calls: one answer, one in-graph audit. The reader receives
only the textual representation of the retrieved graph and its evidence references — no images,
raw video, captions, or gold answers.

- `summary.json`: accuracy, execution failures, per-R-type statistics, in-graph support distribution.
- `results.json`: per-question answers, the actual reader payload, assertion audit, and gaps.
- `calls.json` / `question_calls/`: call latency and token usage.
- `manifest.json`: frozen snapshot, model, access scope, and diagnostic mode.

Under forced answering, an option is chosen even when the evidence is insufficient.
`correct_with_support` stays `null` without an independent media audit; QA accuracy cannot
substitute for binding acceptance.

## 5. Rebuild, visualize, and no-GPU example

```bash
# rebuild from the ledger without re-calling the model; the original ledger stays in its source dir
python tools/rebuild_simple_memory.py \
  --ledger outputs/video01/ledger --out outputs/video01_rebuilt

# HTML of text, carriers, and cited source frames; source frames must be reachable
python tools/render_text_evidence.py \
  --graph outputs/video01/hypergraph.json --out outputs/video01_text
python -m http.server 8767 --directory outputs

# synthetic structure example without a video or model
rrt-echo replay examples/observations.jsonl \
  --identity examples/identity.jsonl --out outputs/toy
rrt-echo inspect outputs/toy/hypergraph.json --question 'Who attacks whom?'
```

After moving source frames across machines, pass `--path-map /old/prefix=/new/prefix` to the text
viewer. The synthetic example only validates structure; `synthetic://` is not real video
evidence. `rrt-echo inspect` is a low-level retrieval example — use `tools/inspect_memory.py` for
the full QA payload.

The interactive hypergraph viewer lives in [`visualization/`](visualization/README.md): it renders
the events, participants, and role bindings over time and traces a QA run back onto the graph.

## 6. Optional ASR

With M3-Agent code and local Whisper/ERes2NetV2 weights:

```bash
python -m pip install -e '.[audio]'
rrt-echo-audio --video /path/to/video.mp4 \
  --m3-root /path/to/m3-agent --device cuda --out outputs/video01_audio
```

Append `--audio-observations outputs/video01_audio/audio.json` to the build command. The visual
model sees time-aligned ASR text; image mode does not listen to audio directly. Voice clusters
are anonymous hypotheses, not automatically visible people; ASR content does not automatically
become world facts.

## 7. Current boundaries

This base has been run end-to-end on one full development video: 45 windows, 349 facts, QA 20/28.
That is not a cross-video generalization metric. OCR, person region/type, cross-shot identity, and
action-initiator roles still have errors; conflict constraints can block wrong merges and can
also cause identity fragmentation.

Videos, models, credentials, experiment outputs, and local history archives do not enter Git.
This is an internal research base; no open-source license is declared.
