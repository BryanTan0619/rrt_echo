#!/usr/bin/env python3
"""Answer BindScope questions with the tool-calling agent over one memory variant.

    export OPENAI_BASE_URL=... OPENAI_API_KEY=...   # any OpenAI-compatible chat endpoint with function calling
    python -m agent.run_agent --memory memory/full --out runs/full [--model deepseek-v4-flash] \
        [--extra-body '{"thinking": {"type": "disabled"}}'] [--videos ID ...] [--workers 8]

<memory>/<video_id>/hypergraph.json is produced by scripts/build_memory.py. Every question gets one JSON
file (full dialogue, every tool call and its output, answer); runs resume. predictions.jsonl and
summary.json are written at the end. --arm blind answers from the question text alone.
"""

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from .memory import Memory, mmss
from .tool import TOOL_SPECS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MAX_TOOL_CHARS = 16000
EXTRA_BODY = None  # set from --extra-body


def chat(cl, **kw):
    if EXTRA_BODY:
        kw["extra_body"] = EXTRA_BODY
    for attempt in range(6):
        try:
            return cl.chat.completions.create(**kw)
        except Exception:  # noqa: BLE001  transient 429 / 5xx
            if attempt == 5:
                raise
            time.sleep(min(60, 3 * 2**attempt))


def video_card(M):
    roles = ", ".join(f"{r}({n})" for r, n in M.role_vocab.most_common(20))
    kinds = Counter(F["kind"] for F in M.facts.values() if not F["dropped"])
    flags = Counter(fl for F in M.facts.values() for fl in F.get("flags", ()))
    return (
        f"Video card: duration {mmss(M.duration)}; {len(M.win_span)} observation windows; facts: {dict(kinds)}; "
        f"{sum(F['dropped'] for F in M.facts.values())} dropped event records available via include_dropped; "
        f"entities: {sum(1 for e in M.ent.values() if e['kind'] == 'person')} person, {sum(1 for e in M.ent.values() if e['kind'] == 'object')} object, "
        f"{sum(1 for e in M.ent.values() if e['kind'] == 'region')} region; speech transcripts: {'yes' if M.has_speech else 'NO (audio was not processed)'}.\n"
        f"Role names used (count): {roles}.\nReliability flag counts: {dict(flags)}."
    )


def normalize_answer(a, q):
    a = str(a or "").strip()
    if q["question_type"] != "MCQ":
        return (
            "True" if a.lower().startswith("t") else ("False" if a.lower().startswith("f") else a)
        )
    m = re.match(r"\(?([A-D])\b", a.upper())
    return m.group(1) if m else a[:1].upper()


def call_tool(M, name, args):
    if name != "search":
        return f"unknown tool {name}"
    try:
        return M.search(**args)
    except Exception as e:  # noqa: BLE001  tool errors are returned to the agent verbatim
        return f"tool error: {type(e).__name__}: {e}"


def run_one(M, q, arm, model, max_steps, cl, prompt):
    qtext = q["question"] + ("\n(Answer True or False.)" if q["question_type"] != "MCQ" else "")
    if arm == "blind":
        msgs = [
            {
                "role": "system",
                "content": "Answer the multiple-choice or true/false question about a video you have NOT seen and have no information about. "
                'Use only the question text and commonsense. Reply with JSON {"answer": <letter or True/False>, "confidence": 0-1, "reasoning": "..."}.',
            },
            {"role": "user", "content": qtext},
        ]
        r = chat(cl, model=model, messages=msgs)
        txt = r.choices[0].message.content or ""
        m = re.search(r'"answer"\s*:\s*"?([A-D]|True|False)', txt)
        return {
            "answer": normalize_answer(m.group(1) if m else txt.strip()[:5], q),
            "raw": txt,
            "steps": 1,
            "tool_calls": 0,
            "usage": [r.usage.prompt_tokens, r.usage.completion_tokens],
            "trace": [],
        }
    valid = {"True", "False"} if q["question_type"] != "MCQ" else {"A", "B", "C", "D"}
    msgs = [
        {"role": "system", "content": prompt + "\n\n" + video_card(M)},
        {"role": "user", "content": qtext},
    ]
    trace, usage, final, n_tools, reminders = [], [0, 0], None, 0, 0
    for step in range(max_steps):
        kw = dict(model=model, messages=msgs, tools=TOOL_SPECS)
        if step == max_steps - 1:
            msgs.append(
                {
                    "role": "user",
                    "content": "Step budget exhausted. Call submit_answer now with your best option.",
                }
            )
            kw["tool_choice"] = {"type": "function", "function": {"name": "submit_answer"}}
        r = chat(cl, **kw)
        usage[0] += r.usage.prompt_tokens
        usage[1] += r.usage.completion_tokens
        m = r.choices[0].message
        am = {"role": "assistant", "content": m.content or ""}
        rc = getattr(m, "reasoning_content", None)
        if rc:
            am["reasoning_content"] = rc  # reasoning models: echo reasoning_content back with tool calls
        if m.tool_calls:
            am["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in m.tool_calls
            ]
        msgs.append(am)
        trace.append(
            {
                "step": step,
                "content": m.content,
                "reasoning": rc,
                "tool_calls": am.get("tool_calls"),
            }
        )
        if not m.tool_calls:
            if reminders < 3:
                reminders += 1
                msgs.append(
                    {
                        "role": "user",
                        "content": "You must end by calling the submit_answer tool (not plain text). Continue searching if needed, then call submit_answer.",
                    }
                )
                continue
            txt = m.content or ""
            mm = (
                re.search(r"\b(true|false)\b", txt, re.I)
                if q["question_type"] != "MCQ"
                else re.search(r"(?:answer|option)\W{0,5}\(?([A-D])\b", txt, re.I)
            )
            final = {
                "answer": mm.group(1) if mm else "",
                "confidence": None,
                "reasoning": "(no submit_answer; parsed from text)",
            }
            break
        # several searches in one turn run concurrently; each result names the call it answers
        calls = []
        for k, tc in enumerate(m.tool_calls, 1):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = {}
            calls.append((k, tc, tc.function.name, args if isinstance(args, dict) else {}))
        todo = [c for c in calls if c[2] == "search"]
        with ThreadPoolExecutor(max(1, len(todo))) as ex:
            outs = dict(
                zip([c[0] for c in todo], ex.map(lambda c: call_tool(M, c[2], dict(c[3])), todo))
            )
        for k, tc, name, args in calls:
            if name == "submit_answer":
                if normalize_answer(args.get("answer"), q) in valid:
                    final, out = args, "submitted"
                else:  # malformed arguments / invalid answer: ask the model to resubmit
                    out = (
                        f"submit_answer rejected: answer must be one of {sorted(valid)}; got {json.dumps(args, ensure_ascii=False)[:200]}. "
                        "Call submit_answer again with a valid JSON object."
                    )
            else:
                n_tools += name == "search"
                out = outs.get(k, f"unknown tool {name}")
            if len(out) > MAX_TOOL_CHARS:
                out = (
                    out[:MAX_TOOL_CHARS]
                    + f"\n… (truncated; {len(out)} chars total — narrow the query or use offset)"
                )
            head = f"[result of call {step + 1}.{k}: {name}({json.dumps(args, ensure_ascii=False)[:300]})]\n"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": head + out})
            trace[-1].setdefault("results", []).append(
                {"call": f"{step + 1}.{k}", "name": name, "args": args, "output": out}
            )
        if final is not None:
            break
    final = final or {"answer": "", "reasoning": "(no answer within step budget)"}
    return {
        "answer": normalize_answer(final.get("answer"), q),
        "confidence": final.get("confidence"),
        "evidence": final.get("evidence"),
        "reasoning": final.get("reasoning"),
        "steps": len(trace),
        "tool_calls": n_tools,
        "usage": usage,
        "trace": trace,
    }


def agent_question(q):
    """BindScope row -> agent input: MCQ options appended as "A) ..." lines; True/False as is."""
    base = {k: q[k] for k in ("question_id", "video_id", "unit_id", "binding_type")}
    if q["question_type"] == "multiple_choice":
        opts = "".join(f"\n{k}) {q['options'][k]}" for k in sorted(q["options"]))
        return {**base, "question_type": "MCQ", "question": q["question"] + opts, "ans": q["answer"]}
    return {**base, "question_type": "True/False", "question": q["question"], "ans": str(q["answer"])}


def main():
    global EXTRA_BODY
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True, help="directory with <video_id>/hypergraph.json")
    ap.add_argument("--questions", default=str(ROOT / "data/bindscope.jsonl"))
    ap.add_argument("--videos", nargs="*", help="default: every video that has a memory file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=["agent", "blind"], default="agent")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--extra-body", default=None, help="JSON merged into every chat request")
    ap.add_argument("--prompt", default=str(HERE / "system_prompt.md"))
    ap.add_argument("--cache", default=str(ROOT / ".cache/embeddings"))
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    EXTRA_BODY = json.loads(a.extra_body) if a.extra_body else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    prompt = Path(a.prompt).read_text()
    Q = [agent_question(json.loads(line)) for line in open(a.questions) if line.strip()]
    root = Path(a.memory)
    videos = a.videos or sorted({q["video_id"] for q in Q if (root / q["video_id"] / "hypergraph.json").exists()})
    mems = {v: Memory(root / v, a.cache) for v in videos} if a.arm == "agent" else {}
    todo = [q for q in Q if q["video_id"] in set(videos)]
    cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"), timeout=180)
    lock = threading.Lock()

    def work(q):
        f = out / (q["question_id"].replace("#", "_").replace(":", "_") + ".json")
        if f.exists():
            return json.loads(f.read_text())
        t = time.time()
        try:
            res = run_one(mems.get(q["video_id"]), q, a.arm, a.model, a.max_steps, cl, prompt)
        except Exception as e:  # noqa: BLE001
            res = {"answer": "", "error": f"{type(e).__name__}: {e}", "trace": []}
        res.update(question_id=q["question_id"], video_id=q["video_id"], binding_type=q["binding_type"],
                   gold=q["ans"], correct=res.get("answer") == q["ans"], seconds=round(time.time() - t, 1))
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        with lock:
            print(f"{q['question_id']} pick={res.get('answer')} gold={q['ans']} {'ok' if res['correct'] else 'x'} "
                  f"tools={res.get('tool_calls')} {res['seconds']}s {res.get('error', '')}", flush=True)
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(work, todo))
    with open(out / "predictions.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps({k: r.get(k) for k in ("question_id", "answer", "gold", "correct", "error")}) + "\n")
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["binding_type"]][0] += r["correct"]
        by_type[r["binding_type"]][1] += 1
    summ = {"arm": a.arm, "model": a.model, "n": len(results), "correct": sum(r["correct"] for r in results),
            "by_type": dict(sorted(by_type.items())), "errors": [r["question_id"] for r in results if r.get("error")]}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
