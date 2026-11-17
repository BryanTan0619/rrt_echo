# ECHO: binding-aware memory for long-video QA

Code and data for the BindScope benchmark and the controlled memory study (RQ3).
Perception is held fixed. Every memory variant is built **forward** from the same cached
observations: VLM window observations, cross-window identity records, and ASR. A single
tool-calling agent then answers the questions over that memory.

```
data/
  bindscope.jsonl                  BindScope: 3,070 questions = 1,535 paired units, 77 videos
  example/<video_id>/observations.json   cached observations for one video (smoke test)
echo/                              memory construction (CPU, standard library only)
  variants.py                      the 12 memory variants of the study (entry point)
  rrt/ledger.py                    observation ledger -> normalised facts, identities, hypergraph
  rrt/memory.py, rrt/hypergraph.py identity admission and the temporal evidence hypergraph
  rrt/event_content.py             co-observation hyperedges (event + co-observed attributes)
  rrt/audio.py                     attaches ASR utterances to the memory
  rrt/representations.py           identity-merge removal, pairwise roles, triple documents
agent/                             tool-calling QA agent (one `search` tool + `submit_answer`)
scripts/
  build_memory.py                  observations -> memory/<variant>/<video_id>/hypergraph.json
  evaluate.py                      QAcc / PairAcc / BIR, per-type tables, video bootstrap, Holm
results/
  predictions/<variant>.jsonl      our per-question answers for every variant
  pair_kind.json                   pair construction type per unit (complementary / paraphrase / tf_minimal)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later. Memory construction and evaluation run offline on CPU. The agent needs an
OpenAI-compatible chat endpoint with function calling. Retrieval embeddings
(all-MiniLM-L6-v2, ONNX) are computed locally. They are downloaded from the Hugging Face Hub on
first use; set `EMBED_MODEL_DIR` to use a local copy.

## 1. Reproduce the tables from our predictions

```bash
python scripts/evaluate.py results/predictions                       # all variants vs. full (ECHO)
python scripts/evaluate.py results/predictions --subset complementary
python scripts/evaluate.py results/predictions --latex                # rows in the paper's table layout
```

The denominator is fixed to all 1,535 paired units: a missing answer counts as wrong. Differences are
bootstrapped over videos (10,000 resamples). Holm correction runs over the files you pass, so pass
only the variants of one table to reproduce that table's adjusted p-values.

## 2. Build the memories

```bash
python scripts/build_memory.py --observations data/example --out memory
```

| variant | memory | paper name |
|---|---|---|
| `full` | ECHO hypergraph | ECHO |
| `binary` | binary graph: events split into pairwise links; attributes, states, text as separate links | Binary graph |
| `triple` | LightRAG-style triples, repeated occurrences merged | Triple |
| `flat` | template captions, one passage per observation window | Flat text |
| `no_identity` | hypergraph without cross-window identity merges | w/o identity merging |
| `no_temporal` | hypergraph with no time | w/o temporal scope |
| `no_identity_no_temporal` | both | w/o identity & temporal |
| `full_30s`, `binary_30s` | the same memories on a 30-s clock | (30 s) |
| `binary_untimed`, `triple_untimed`, `flat_untimed` | the same memories with no time | (untimed) |

The full set of cached observations (77 videos) is distributed separately. Put each video at
`<dir>/<video_id>/observations.json` and pass `--observations <dir>`.

## 3. Run the agent

```bash
export OPENAI_BASE_URL=<endpoint>  OPENAI_API_KEY=<key>
python -m agent.run_agent --memory memory/full --out runs/full \
    --model <model name> --extra-body '{"thinking": {"type": "disabled"}}'
python scripts/evaluate.py runs/full/predictions.jsonl runs/binary/predictions.jsonl --videos <video_id ...>
```

Our runs used DeepSeek-V4-Flash with thinking disabled, at most 40 agent steps per question, and
the same prompt, tools and embedding model for every variant. `--extra-body` passes
provider-specific options through unchanged. Each question writes one JSON file with the full
dialogue and every tool call; interrupted runs resume. `--arm blind` answers from the question text
alone.

## BindScope format

One question per line: `question_id`, `video_id`, `unit_id` (two questions per paired unit),
`binding_type` (R0–R7), `question_type` (`multiple_choice` or `true_false`), `question`, `options`
(A–D or True/False), and `answer`. A unit is correct when both of its questions are correct (PairAcc).
BIR is the share of units with exactly one correct probe among units with at least one.
