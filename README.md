# ECHO: binding-preserving memory for long-video QA

ECHO stores what a video shows as a temporal evidence hypergraph: every observed event is one
hyperedge that keeps its participants, their roles, the attributes observed with them, and the
time, while identities are linked across windows only through accepted identity decisions. At
query time a tool-calling agent searches this memory with explicit scopes (time, entity, keyword,
window), and every hit comes back with its whole binding unit.

This repository contains the memory construction, the agent, the BindScope benchmark, and cached
observations for one example video.

```
data/
  bindscope.jsonl                          BindScope: 3,070 questions = 1,535 paired units, 77 videos
  example/<video_id>/observations.json     cached observations for one video
echo/                                      memory construction (CPU, standard library only)
  rrt/ledger.py                            observations -> ECHO memory (entry point: build_memory)
  rrt/memory.py, rrt/hypergraph.py         identity admission and the temporal evidence hypergraph
  rrt/event_content.py                     co-observation hyperedges (event + co-observed attributes)
  rrt/audio.py                             attaches ASR utterances to the memory
agent/                                     tool-calling agent: one `search` tool + `submit_answer`
scripts/build_memory.py                    data/example -> memory/<video_id>/hypergraph.json
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Memory construction runs offline on CPU. The agent needs an OpenAI-compatible
chat endpoint with function calling. Retrieval embeddings (all-MiniLM-L6-v2, ONNX) are computed
locally. They are downloaded from the Hugging Face Hub on first use; set `EMBED_MODEL_DIR` to use a
local copy.

## Build the memory

```bash
python scripts/build_memory.py --observations data/example --out memory
```

Each `observations.json` holds the cached perception of one video: VLM observations per window, the
cross-window identity journal, and ASR utterances. The observations for all 77 videos are
distributed separately; place each at `<dir>/<video_id>/observations.json` and pass
`--observations <dir>`.

## Answer questions

```bash
export OPENAI_BASE_URL=<endpoint>  OPENAI_API_KEY=<key>
python -m agent.run_agent --memory memory --out runs/echo \
    --model <model name> --extra-body '{"thinking": {"type": "disabled"}}'
```

We used DeepSeek-V4-Flash with thinking disabled and at most 40 agent steps per question.
`--extra-body` passes provider-specific options through unchanged. Each question writes one JSON
file with the full dialogue and every tool call; interrupted runs resume. `predictions.jsonl` lists
`question_id`, `answer`, `gold`, and `correct` for every question.

## BindScope

One question per line: `question_id`, `video_id`, `unit_id` (two questions per paired unit),
`binding_type` (R0–R7), `question_type` (`multiple_choice` or `true_false`), `question`, `options`
(A–D or True/False), and `answer`. A unit counts as correct when both of its questions are correct
(PairAcc). BIR is the share of units with exactly one correct question among units with at least one.
