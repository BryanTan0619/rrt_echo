# Hypergraph viewer

Renders a finished RRT-ECHO run as an interactive page — events, participants, and role
bindings unfolding over time — and traces a `rrt.reader` QA run back onto the graph.

**Read-only.** It reads `hypergraph.json` and the other artifacts and renders them; it never
writes back, never merges entities, never rewrites predicates, and never translates. Every
event, entity, and role shown is exactly what the artifact stores.

## Install

The page itself needs no install (it runs D3 in the browser, fetched from a CDN). Building the
data needs:

```bash
python -m pip install pillow   # avatars are cropped from source frames with the pipeline's reference boxes
```

`ffmpeg` is optional — it only adds a WebM fallback for the preview video; if missing, that step
is skipped.

## Three steps for your own run

```bash
# 1. build page data from a finished run
python visualization/hypergraph_viewer/build_data.py <run dir> "<display name>" \
    --path-map /old/machine/prefix/=/this/machine/prefix/   # only when crossing machines; repeatable

# 2. (optional) attach a reader QA run so answers trace back to the graph
python visualization/hypergraph_viewer/build_qa.py <qa dir> "<display name>" <questions jsonl> [annotation globs ...]

# 3. serve and open
python visualization/hypergraph_viewer/serve.py 8109
```

`<display name>` must match between steps 1 and 2. Data defaults to
`visualization/hypergraph_viewer/runs/` (override with `--out`). **That directory is an
experiment artifact — do not commit it.**

## What is drawn

One page (`index.html`): an event is a diamond, its participants are circles, and a role line
runs from the diamond to each participant with the role written on it. Events are not enclosed
in outlines; the canvas is split into cells (Voronoi regions around the diamonds), close to the
paper's Figure 1. A person who participates in several events tends to fall near the boundary
between their cells — a force-layout result, not a hard constraint.

| Click | Effect |
|---|---|
| a diamond (event) | select it; the video jumps to its time |
| a person / object | select it, see all its events and attributes; **the playhead stays** unless it has exactly one event |
| a time in the left list | jump to that time |
| empty canvas | deselect, and return to where you were before the jump |

Selecting dims everything else, leaving only what is related.

**Time.** The gray band on the time bar is the look-back window: only events inside it are
drawn, older ones are more transparent (tunable with the "past fade" slider down to never).
The window size is set by the "look-back window" slider, or by dragging the band's start
handle; it follows the playhead while playing. "not-yet opacity" controls whether not-yet
events are drawn (default 0 = hidden).

**View.** Wheel to zoom, drag empty space to pan, double-click to reset. When zoomed in, the
view follows the event happening now.

## QA trace

After step 2, a "QA trace" switch appears at the top. Turning it on lists each question — the
reader's pick, the gold answer, the number of retrieved facts, and the audit verdict. Clicking
a question leaves only the facts the reader was shown on the graph (everything else hidden,
the time window ignored, because the reader sees the whole memory):

- magenta ticks = the times of those facts; black ticks / thick borders = facts the audit cited;
- each fact is tagged by which retrieval hypothesis pulled it in (the stem, or only an option);
- if the question file carries provenance, the gold evidence span and each distractor's time
  appear as chips; opening one shows **what the memory actually recorded in that span**, with
  how many of those facts were retrieved.

That last view is diagnostic: it separates "not recorded", "recorded but not retrieved", and
"retrieved but answered wrongly".

## Caveats

1. **Artifacts store the absolute paths of the machine that produced them.** Crossing machines
   needs `--path-map`, otherwise avatars and the preview video cannot be found (the page still
   draws, just without pictures).
2. **Avatars are cropped from source frames** using `entity_registry.json` reference boxes; no
   source frames, no avatars. A person node's label is `first-local-instance@time (+N)`, where
   `+N` is the number of extra local instances linked into that entity.
3. **Instance descriptions are not rewritten into attributes.** If a description says
   "…with number 01 on her chest" but there is no corresponding text fact, the panel shows "no
   attributes recorded" — the number lives only in the description. This is deliberate; the page
   does not complete or promote anything.
4. **`build_qa.py` has no dataset-specific rules.** The gold answer comes from the question file
   you pass in; the time spans come from the optional annotation files; with neither, it only
   shows the answer and the audit.
5. **Page data embeds base64 avatars** (~4 MB for a 25-minute run); use `--no-avatars` to share.
6. The browser needs access to `cdn.jsdelivr.net` for D3; on an offline network, vendor D3
   locally.
7. `serve.py` supports Range requests (required for video seeking) and binds `0.0.0.0` — only
   run it on a trusted network.

## The other file: `vllm_compat_proxy.py`

Unrelated to the viewer; the shim we used to run the pipeline against vLLM 0.19.1. See the
top-level `README.md` for the note.
