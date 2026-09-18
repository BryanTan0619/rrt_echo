#!/usr/bin/env python3
"""Attach a reader's QA output (rrt_echo.rrt.reader --diagnostic) to a viz run, so each question can be traced back
onto the hypergraph.

Reads only author artefacts, writes runs/<name>/qa.json. Nothing in the author's tree is touched.

    python build_qa.py <qa dir> <run name> <questions jsonl> [annotation glob ...] [--out DIR]

<qa dir> is the output of `rrt-echo-qa` / rrt.reader (results.json, summary.json, manifest.json, question_calls/).
<questions jsonl> is the same question file that was answered. Optional annotation globs add whatever the question
set records about where the answer is in the video; nothing is required and no answer rule is built in.

Per question we keep: stem + options, gold letter (from the user's jsonl), the reader's letter, correct?, the audit
verdict (support_verdict / assertions with their evidence_chain), and every fact the reader was shown (reader_payload.facts):
its kind / predicate / value / extent, the entity ids behind its roles (resolved_roles), which retrieval hypothesis pulled
it in (retrieval_hypotheses: question / option_A…), and whether the audit cited it (evidence_chain).
Event facts share their id with the page's bubbles (build_data.py uses fact_id as bubble id), so the page can light them up.
"""

import argparse
import glob
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser(
    description="Attach a reader run to a viewer run so answers trace back to the graph."
)
ap.add_argument(
    "qa_dir", help="reader output directory (results.json, summary.json, question_calls/)"
)
ap.add_argument("name", help="viewer run name, as passed to build_data.py")
ap.add_argument("questions", help="the questions jsonl that was answered")
ap.add_argument(
    "annotations", nargs="*", help="optional globs of jsonl carrying per-question provenance"
)
ap.add_argument(
    "--out", default=None, help="viewer runs/ directory (default: runs/ next to the pages)"
)
_a = ap.parse_args()
qa_dir, run_name, gold_path = Path(_a.qa_dir), _a.name, Path(_a.questions)
src_globs = _a.annotations


def hhmmss(text):
    """'16:18-16:20' / '17:08–17:14' / '04:43' -> (start, end) in seconds, or None."""
    parts = re.findall(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})", str(text))
    if not parts:
        return None
    secs = [int(h or 0) * 3600 + int(m) * 60 + int(s) for h, m, s in parts]
    return (secs[0], secs[1]) if len(secs) > 1 and secs[1] >= secs[0] else (secs[0], secs[0])


def norm_q(text):
    return re.sub(r"\s+", " ", str(text).split("\n")[0].strip().lower())


ann = {}
for g in src_globs:
    for f in glob.glob(g):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                ann.setdefault(norm_q(row.get("question", "")), (Path(f).name, row))
out_dir = (
    Path(_a.out).resolve() if _a.out else Path(__file__).resolve().parent / "runs"
) / run_name
data = json.loads((out_dir / "data.json").read_text())
bubble_ids = {b["id"] for b in data["views"]["graph"]["bubbles"]}
ent_keys = {e["key"] for e in data["views"]["graph"]["entities"]}

results = json.loads((qa_dir / "results.json").read_text())
summary = json.loads((qa_dir / "summary.json").read_text())
manifest = json.loads((qa_dir / "manifest.json").read_text())
gold = {}
for i, line in enumerate(gold_path.read_text().splitlines()):
    if not line.strip():
        continue
    g = json.loads(line)
    gold[g.get("question_id") or f"q{i + 1:02d}"] = g


def split_question(text):
    lines = text.split("\n")
    stem, opts = [], {}
    for ln in lines:
        s = ln.strip()
        if len(s) > 2 and s[0] in "ABCDEFGH" and s[1] == ")":
            opts[s[0]] = s[2:].strip()
        else:
            stem.append(s)
    return " ".join(x for x in stem if x), opts


questions = []
for i, r in enumerate(results):
    qid = r["question_id"]
    g = gold.get(qid, {})
    stem, opts = split_question(g.get("question", ""))
    pl = r["reader_payload"]
    hyp = pl.get("retrieval_hypotheses") or {}
    pulled = {}
    for k, ids in hyp.items():
        tag = "stem" if k == "question" else k.replace("option_", "")
        for fid in ids:
            pulled.setdefault(fid, []).append(tag)
    chain = set()
    for a in r.get("assertions") or []:
        chain.update(a.get("evidence_chain") or [])
    facts = []
    for f in pl.get("facts") or []:
        rr = f.get("resolved_roles") or {}
        members = sorted(
            {v["entity_id"] for v in rr.values() if isinstance(v, dict) and v.get("entity_id")}
        )
        ext = f.get("extent") or (f.get("observed_times") or [None, None])[:1] * 2
        facts.append(
            {
                "id": f["fact_id"],
                "kind": f.get("kind"),
                "pred": f.get("predicate"),
                "value": f.get("value"),
                "roles": {
                    slot: (rr.get(slot) or {}).get("entity_id") for slot in (f.get("roles") or {})
                },
                "members": members,
                "t0": ext[0] if ext else None,
                "t1": ext[-1] if ext else None,
                "in": pulled.get(f["fact_id"], []),
                "chain": f["fact_id"] in chain,
                "drawn": f["fact_id"] in bubble_ids,
                "members_drawn": [m for m in members if m in ent_keys],
                "support": (f.get("support") or {}).get("local_joint", {}).get("status"),
            }
        )
    facts.sort(key=lambda x: (x["t0"] is None, x["t0"] or 0))
    calls = []
    cp = qa_dir / "question_calls" / f"{i:03d}.json"
    if cp.exists():
        for c in json.loads(cp.read_text()):
            u = c.get("usage") or {}
            calls.append(
                {
                    "prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "elapsed": round(c.get("elapsed_seconds") or 0, 1),
                    "reply": ((c.get("response") or {}).get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")[:400],
                }
            )
    ts = [x for f in facts for x in (f["t0"], f["t1"]) if x is not None]
    src_name, a = ann.get(norm_q(g.get("question", "")), (None, {}))
    gold_spans = [list(x) for x in (hhmmss(t) for t in (a.get("provenance") or [])) if x]
    # Distractor times come straight from the delivered wrong_options text: options were
    # shuffled on delivery, so source-file letters do not match delivered letters.
    opt_span = {}
    for L, text in (g.get("wrong_options") or {}).items():
        sp = hhmmss(text)
        if sp:
            opt_span[L] = list(sp)
    events = []
    for key, label in (("query_event", "query event"), ("bridge_event", "bridge event")):
        if a.get(key):
            sp = hhmmss(a[key])
            events.append(
                {
                    "kind": label,
                    "text": re.sub(r"^[\d:\s]*[—–-]\s*", "", str(a[key])),
                    "span": list(sp) if sp else None,
                }
            )
    annotation = (
        {
            "source": src_name,
            "qa_id": a.get("qa_id"),
            "gold_spans": gold_spans,
            "target_window": list(hhmmss(a.get("target_window")))
            if a.get("target_window")
            else None,
            "option_spans": opt_span,
            "events": events,
            "binding_category": a.get("binding_category"),
            "memory_dependency": a.get("memory_dependency"),
            "modality": a.get("modality"),
            "bridge_target": a.get("bridge_target"),
            "bridge_gap_sec": a.get("bridge_gap_sec"),
            "audio_leak": a.get("audio_leak"),
        }
        if a
        else None
    )
    questions.append(
        {
            "id": qid,
            "type": r.get("type"),
            "level": g.get("level"),
            "stem": stem,
            "options": opts,
            "annotation": annotation,
            "gold": g.get("ans"),
            "answer": r.get("answer"),
            "correct": bool(r.get("correct")),
            "wrong_options": g.get("wrong_options") or {},
            "support_verdict": r.get("support_verdict"),
            "scope": r.get("scope"),
            "top_k": r.get("retrieval_top_k"),
            "assertions": [
                {
                    "text": a.get("assertion"),
                    "claim_type": a.get("claim_type"),
                    "verdict": a.get("support_verdict"),
                    "chain": a.get("evidence_chain") or [],
                }
                for a in (r.get("assertions") or [])
            ],
            "facts": facts,
            "span": [min(ts), max(ts)] if ts else None,
            "n_event": sum(f["kind"] == "event" for f in facts),
            "n_other": sum(f["kind"] != "event" for f in facts),
            "closure_complete": pl.get("closure_complete"),
            "delivery_complete": pl.get("delivery_complete"),
            "omitted_for_budget": len(pl.get("omitted_for_budget") or []),
            "calls": calls,
        }
    )

out = {
    "source": str(qa_dir),
    "model": manifest.get("model"),
    "diagnostic": manifest.get("diagnostic"),
    "snapshot_id": manifest.get("snapshot_id"),
    "gold_file": str(gold_path),
    "summary": {
        "questions": summary.get("questions"),
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "answer_support": summary.get("answer_support"),
        "by_type": {
            k: {"questions": v["questions"], "correct": v["correct"]}
            for k, v in (summary.get("by_type") or {}).items()
        },
    },
    "questions": questions,
}
(out_dir / "qa.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(
    f"wrote {out_dir / 'qa.json'}: {len(questions)} questions, correct {out['summary']['correct']}, "
    f"facts drawn {sum(f['drawn'] for q in questions for f in q['facts'])}/{sum(len(q['facts']) for q in questions)}"
)
print(f"annotation matched {sum(bool(q['annotation']) for q in questions)}/{len(questions)}")
for q in questions:
    ga = q["annotation"] or {}
    print(
        q["id"],
        q["type"],
        "✓" if q["correct"] else "✗",
        q["answer"],
        "gold",
        q["gold"],
        "events",
        q["n_event"],
        "| gold spans",
        [
            f"{s0 // 60}:{s0 % 60:02d}-{s1 // 60}:{s1 % 60:02d}"
            for s0, s1 in ga.get("gold_spans", [])
        ],
        "| facts falling in gold spans:",
        sum(
            1
            for f in q["facts"]
            for s0, s1 in ga.get("gold_spans", [])
            if f["t0"] is not None and f["t0"] <= s1 and f["t1"] >= s0
        ),
    )
