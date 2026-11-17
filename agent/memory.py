"""Retrieval layer over one memory file (<run_dir>/hypergraph.json): builds the index behind the
agent's single retrieval tool `search` (see tool.py, SEARCH_SPEC). Read-only."""

import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def norm_phrase(text):
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def parse_time(v):
    """Accept seconds, "mm:ss", "h:mm:ss" or "mm:ss.s"."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    parts = str(v).strip().split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        return None
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def mmss(t):
    if t is None:
        return "?"
    return f"{int(t // 60)}:{t % 60:04.1f}"


TARGET_ROLES = {
    "patient",
    "held",
    "recipient",
    "receiver",
    "listener",
    "passenger",
    "pointee",
    "shown",
    "target",
    "addressee",
    "object",
    "item",
    "theme",
    "victim",
    "carried",
    "worn",
    "used",
    "eaten",
    "watched",
    "viewed",
}


CONTEXT_ROLES = {
    "instrument",
    "surface",
    "scene",
    "viewpoint",
    "displayed content",
    "location",
    "container",
    "tool",
    "vehicle",
    "background",
    "place",
    "destination",
    "source",
}


OWNER_ROLES = {"owner"}


def role_group(r):
    r = r.casefold()
    if r in OWNER_ROLES:
        return "@owner"
    if r in TARGET_ROLES:
        return "@target"
    if r in CONTEXT_ROLES:
        return "@context"
    return "@actor"  # agent/holder/speaker/writer/driver/observer/singer/...: default to the initiating side


FLAG_TEXT = {
    "roles_repeated_in_window": "≥3 different events in this window share exactly the same role assignment — the model probably copied a role template; roles unreliable",
    "object_in_actor_role": "an object fills a doer role",
    "person_in_context_role": "a person fills an instrument/surface role",
    "shared_box_audit": "original audit: different roles use the same box, or one box covers several people — who-is-who unclear",
    "unresolved_slots": "the model marked some slots as unresolved",
    "copied_boxes": "a participant's box is identical in every frame (probably copied) — may point at the wrong thing after a cut",
    "whole_frame_box": "a participant's box is the whole frame — not localized",
    "overlay_owner_guessed": "post-production caption; the owner is the model's guess, not necessarily the speaker",
    "single_window_identity": "a participant appears in only one window (not linked across windows); the same person may exist under other ids",
    "dropped_by_validator": "dropped by the original validator (see reason); weak evidence only",
    "speaker_is_voice_cluster": "transcribed speech; speaker is an anonymous voice cluster, not a visible person; content is what was said, not what happened",
}


class Embedder:
    """all-MiniLM-L6-v2 sentence embeddings (ONNX export; mean pooling, L2-normalised, 384-d), run locally.

    The model files are fetched once from the Hugging Face Hub (or read from $EMBED_MODEL_DIR).
    Vectors are cached in memory and in <cache_dir>/emb_minilm.jsonl."""

    REPO = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, cache_dir, dim=384):
        self.dim = dim
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache_path = Path(cache_dir) / "emb_minilm.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.open():
                k, v = json.loads(line)
                self.cache[k] = v
        self._model = None
        self._lock = threading.Lock()

    def model(self):
        if self._model is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            d = os.environ.get("EMBED_MODEL_DIR")
            if d:
                onnx_path, tok_path = Path(d) / "onnx/model.onnx", Path(d) / "tokenizer.json"
            else:
                from huggingface_hub import hf_hub_download

                onnx_path = hf_hub_download(self.REPO, "onnx/model.onnx")
                tok_path = hf_hub_download(self.REPO, "tokenizer.json")
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("EMBED_THREADS", "4"))
            sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
            tok = Tokenizer.from_file(str(tok_path))
            tok.enable_truncation(256)
            tok.enable_padding()
            self._model = (sess, tok)
        return self._model

    def encode(self, texts):
        sess, tok = self.model()
        names = {i.name for i in sess.get_inputs()}
        out = []
        for i in range(0, len(texts), 128):
            enc = tok.encode_batch([t[:2000] or " " for t in texts[i : i + 128]])
            ids = np.array([e.ids for e in enc], np.int64)
            mask = np.array([e.attention_mask for e in enc], np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in names:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = sess.run(None, feed)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def embed(self, texts):
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        with self._lock:
            todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in self.cache})
            if todo:
                vecs = self.encode([t for _, t in todo])
                with self.cache_path.open("a") as fh:
                    for (k, _), v in zip(todo, vecs):
                        self.cache[k] = v.tolist()
                        fh.write(json.dumps([k, self.cache[k]]) + "\n")
            m = np.array([self.cache[k] for k in keys], dtype=np.float32).reshape(len(keys), self.dim)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
        return m


def _load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


class Memory:
    def __init__(self, run_dir, cache_dir, embed=True, audio_json=None):
        run = Path(run_dir)
        self.run = run
        h = _load(run / "hypergraph.json")
        obs = _load(run / "observations.json", []) or []
        cov = {
            c["clip_id"]: c["target"]
            for c in (_load(run / "coverage.json", {}) or {}).get("intervals", [])
        }

        # window numbers, ordered by the start of their target interval (only when observations.json is present)
        wins = []
        for o in obs:
            oid = o["observation"]["observation_id"]
            t = cov.get(o.get("input_clip_id")) or cov.get(oid) or [None, None]
            wins.append((t[0] if t[0] is not None else 1e9, oid, t))
        wins.sort()
        self.win_of = {oid: i + 1 for i, (_, oid, _) in enumerate(wins)}
        self.win_span = {i + 1: t for i, (_, _, t) in enumerate(wins)}
        obs_by_id = {o["observation"]["observation_id"]: o for o in obs}
        self.obs_by_id = obs_by_id
        self.oid_of_win = {w: oid for oid, w in self.win_of.items()}

        self.instances = {i["instance_id"]: i for i in h["instances"]}
        ent_of = {}
        for e in h["entities"]:
            for li in e["local_instances"]:
                ent_of[li] = e["entity_id"]
        self.ent_of = ent_of

        # box quality of each local instance
        self.inst_flags = {}
        for iid, i in self.instances.items():
            boxes = [tuple(r["box"]) for r in i.get("regions", [])]
            f = set()
            if len(boxes) >= 2 and len(set(boxes)) == 1:
                f.add("copied_boxes")
            if any(
                b[0] <= 0.001 and b[1] <= 0.001 and b[2] >= 0.999 and b[3] >= 0.999 for b in boxes
            ):
                f.add("whole_frame_box")
            self.inst_flags[iid] = f

        facts = [f for f in h["facts"] if not f.get("retrieval_quarantined")]
        # entities: aliases P/O/R in order of first appearance
        first_seen = {}
        for f in facts:
            for rr in f.get("resolved_roles", {}).values():
                e = rr["entity_id"]
                t = min(f["observed_times"] or [1e9])
                first_seen[e] = min(first_seen.get(e, 1e9), t)
        for e in h["entities"]:
            if e["entity_id"] not in first_seen:
                ts = [
                    s
                    for li in e["local_instances"]
                    for s in (self.instances[li].get("extent") or [])
                ]
                first_seen[e["entity_id"]] = min(ts) if ts else 1e9
        kinds = {}
        for e in h["entities"]:
            kc = Counter(
                self.instances[li]["kind"] for li in e["local_instances"] if li in self.instances
            )
            kinds[e["entity_id"]] = kc.most_common(1)[0][0] if kc else "object"
        prefix = {"person": "P", "object": "O", "region": "R"}
        counters = Counter()
        self.alias, self.ent = {}, {}
        for eid in sorted(kinds, key=lambda x: (first_seen.get(x, 1e9), x)):
            p = prefix.get(kinds[eid], "O")
            counters[p] += 1
            a = f"{p}{counters[p]}"
            self.alias[eid] = a
        ents = {e["entity_id"]: e for e in h["entities"]}
        for eid, a in self.alias.items():
            e = ents[eid]
            mem = [self.instances[li] for li in e["local_instances"] if li in self.instances]
            self.ent[a] = {
                "alias": a,
                "entity_id": eid,
                "kind": kinds[eid],
                "instances": [m["instance_id"] for m in mem],
                "descs": [
                    (
                        self._win_of_inst(m["instance_id"]),
                        m.get("description", ""),
                        (m.get("extent") or [None])[0],
                        [
                            f"{x.get('dimension')}: {x.get('value')}"
                            for x in (m.get("attributes") or [])
                            if isinstance(x, dict)
                        ],
                    )
                    for m in sorted(mem, key=lambda m: (m.get("extent") or [1e9])[0])
                ],
                "linked": e.get("status") == "linked",
                "facts": [],
                "texts": [],
                "first": None,
                "last": None,
            }

        # facts
        self.facts = {}
        order = sorted(facts, key=lambda f: (min(f["observed_times"] or [1e9]), f["fact_id"]))
        audits = set()
        for x in _load(run / "binding_loss_audit.json", []) or []:
            audits.add(x["fact_id"])
        for x in h.get("endpoint_grounding_gaps", []):
            audits.add(x["fact_id"])
        for n, f in enumerate(order, 1):
            fid = f"F{n}"
            oid = f["fact_id"].rsplit(":", 2)[0]
            times = f["observed_times"] or []
            roles = []
            for r, li in f["roles"].items():
                eid = (f.get("resolved_roles", {}).get(r) or {}).get("entity_id") or ent_of.get(li)
                a = self.alias.get(eid)
                inst = self.instances.get(li, {})
                roles.append(
                    {
                        "role": r,
                        "group": role_group(r),
                        "entity": a,
                        "kind": inst.get("kind"),
                        "desc": inst.get("description", ""),
                        "inst": li,
                    }
                )
            rec = {
                "id": fid,
                "raw_id": f["fact_id"],
                "kind": f["kind"],
                "predicate": f.get("predicate") or "",
                "value": f.get("value"),
                "carrier": f.get("carrier"),
                "roles": roles,
                "t0": min(times) if times else None,
                "t1": max(times) if times else None,
                "window": self.win_of.get(oid),
                "obs": oid,
                "unresolved": f.get("unresolved_slots") or [],
                "dropped": False,
                "audit": f["fact_id"] in audits,
            }
            self.facts[fid] = rec
            for ro in roles:
                if ro["entity"] in self.ent:
                    E = self.ent[ro["entity"]]
                    if not E["facts"] or E["facts"][-1] != fid:
                        E["facts"].append(fid)
                    if rec["t0"] is not None:
                        E["first"] = rec["t0"] if E["first"] is None else min(E["first"], rec["t0"])
                        E["last"] = rec["t1"] if E["last"] is None else max(E["last"], rec["t1"])
                    if f["kind"] in ("text", "attribute", "state") and ro["role"] == "owner":
                        E["texts"].append(fid)

        # event records rejected by the observation validator (events only: predicate + roles)
        n = 0
        for o in obs:
            oid = o["observation"]["observation_id"]
            regions = {
                s["region_id"].rsplit(":", 1)[-1]: s["local_instance"]
                for s in o.get("spatial_observations", [])
            }
            for rj in o.get("rejected_records", []):
                rec = rj.get("record") or {}
                if not rec.get("predicate"):
                    continue
                n += 1
                roles = []
                for ro in rec.get("roles", []) if isinstance(rec.get("roles"), list) else []:
                    li = regions.get(str(ro.get("region")))
                    eid = ent_of.get(li)
                    inst = self.instances.get(li, {})
                    roles.append(
                        {
                            "role": ro.get("role", "?"),
                            "group": role_group(ro.get("role", "?")),
                            "entity": self.alias.get(eid),
                            "kind": inst.get("kind"),
                            "desc": inst.get("description", ""),
                            "inst": li,
                        }
                    )
                w = self.win_of.get(oid)
                span = self.win_span.get(w) or [None, None]
                did = f"D{n}"
                self.facts[did] = {
                    "id": did,
                    "raw_id": None,
                    "kind": "event",
                    "predicate": rec["predicate"],
                    "value": None,
                    "carrier": None,
                    "roles": roles,
                    "t0": span[0],
                    "t1": span[1],
                    "window": w,
                    "obs": oid,
                    "unresolved": rec.get("unresolved_slots") or [],
                    "dropped": True,
                    "reason": rj.get("reason"),
                    "audit": False,
                }

        # speech: audio_memory.utterances; the speaker is an anonymous voice cluster, not a visible person
        audio = h.get("audio_memory") or {}
        if not audio.get("utterances") and audio_json and Path(audio_json).exists():
            # optional external audio.json with the same utterance format
            audio = _load(audio_json, {}) or {}
        voice = {}
        for n, u in enumerate(
            sorted(audio.get("utterances", []), key=lambda u: u.get("start", 0)), 1
        ):
            spk = str(u.get("speaker_id"))
            voice.setdefault(spk, f"V{len(voice) + 1}")
            sid = f"S{n}"
            self.facts[sid] = {
                "id": sid,
                "raw_id": u.get("utterance_id"),
                "kind": "speech",
                "predicate": "says",
                "value": u.get("text", ""),
                "carrier": "asr",
                "roles": [
                    {
                        "role": "speaker",
                        "group": "@actor",
                        "entity": voice[spk],
                        "kind": "voice",
                        "desc": f"anonymous voice cluster {voice[spk]}",
                        "inst": None,
                    }
                ],
                "t0": u.get("start"),
                "t1": u.get("end"),
                "window": None,
                "obs": None,
                "unresolved": [],
                "dropped": False,
                "audit": False,
                "flags": {"speaker_is_voice_cluster"},
            }
        self.has_speech = bool(voice)

        # reliability flags
        by_win = defaultdict(list)
        for F in self.facts.values():
            if F["kind"] == "event" and not F["dropped"]:
                by_win[F["window"]].append(F)
        for w, lst in by_win.items():
            sig = Counter(
                tuple(sorted((r["role"], r["entity"]) for r in F["roles"]))
                for F in lst
                if len(F["roles"]) >= 2
            )
            for F in lst:
                s = tuple(sorted((r["role"], r["entity"]) for r in F["roles"]))
                preds = {
                    G["predicate"]
                    for G in lst
                    if tuple(sorted((r["role"], r["entity"]) for r in G["roles"])) == s
                }
                if len(F["roles"]) >= 2 and sig[s] >= 3 and len(preds) >= 3:
                    F.setdefault("flags", set()).add("roles_repeated_in_window")
        for F in self.facts.values():
            fl = F.setdefault("flags", set())
            if F["dropped"]:
                fl.add("dropped_by_validator")
            if F["audit"]:
                fl.add("shared_box_audit")
            if F["unresolved"]:
                fl.add("unresolved_slots")
            if F["kind"] == "text" and F.get("carrier") == "overlay":
                fl.add("overlay_owner_guessed")
            for r in F["roles"]:
                if r["group"] == "@actor" and r["kind"] == "object":
                    fl.add("object_in_actor_role")
                if r["group"] == "@context" and r["kind"] == "person":
                    fl.add("person_in_context_role")
                fl |= self.inst_flags.get(r["inst"], set())
                E = self.ent.get(r["entity"])
                if E and len({d[0] for d in E["descs"]}) <= 1 and E["kind"] == "person":
                    fl.add("single_window_identity")

        # identity evidence: per-pair verdicts and whether the memory accepted them (optional files)
        self.idlinks = defaultdict(list)  # alias -> [(other alias, verdict, status/why, basis)]
        for L in h.get("binding_links", []):
            if L.get("relation") != "same_identity" or not str(L.get("proposal_id", "")).startswith(
                "association:"
            ):
                continue
            a, b = self.alias.get(ent_of.get(L["source"])), self.alias.get(ent_of.get(L["target"]))
            if not a or not b or a == b and L["status"] == "accepted":
                continue
            v = {"supported": "same", "contradicted": "different"}.get(
                L.get("verdict"), L.get("verdict")
            )
            why = L.get("status") + (
                "/" + L["acceptance_basis"] if L.get("acceptance_basis") else ""
            )
            self.idlinks[a].append((b, v, why, L.get("basis", "")))
            self.idlinks[b].append((a, v, why, L.get("basis", "")))
        # identity answers rejected by the validator (association/calls/*/rejected_rows.json, optional):
        # rows carry no instance ids, so they are resolved through pair_id in the same batch request.json.
        cdir = run / "association" / "calls"
        rej_rows = {}
        if cdir.exists():
            call_dirs = list(cdir.iterdir())
            with ThreadPoolExecutor(32) as ex:
                for rows in ex.map(lambda d: _load(d / "rejected_rows.json", []) or [], call_dirs):
                    for x in rows:
                        for row in x.get("rows", []):
                            rej_rows[x["pair_id"]] = (
                                row.get("verdict"),
                                row.get("basis"),
                                x.get("error"),
                            )
            if rej_rows:
                with ThreadPoolExecutor(32) as ex:
                    reqs = list(ex.map(lambda d: _load(d / "request.json", {}) or {}, call_dirs))
                for req in reqs:
                    for p in req.get("pairs", []):
                        if p["pair_id"] in rej_rows:
                            v, basis, err = rej_rows[p["pair_id"]]
                            a = self.alias.get(ent_of.get(p["left"]))
                            b = self.alias.get(ent_of.get(p["right"]))
                            if a and b and a != b:
                                self.idlinks[a].append(
                                    (b, v, f"answer_rejected/{err}", basis or "")
                                )
                                self.idlinks[b].append(
                                    (a, v, f"answer_rejected/{err}", basis or "")
                                )

        # co-observation hyperedges (event_hyperedges): an event and the attributes / text observed on the same evidence frames form one group that retrieval returns together
        raw2id = {F["raw_id"]: F["id"] for F in self.facts.values() if F["raw_id"]}
        self.raw2id = raw2id
        self.coobs = defaultdict(set)
        for e in h.get("event_hyperedges", []):
            ids = [raw2id[x] for x in e.get("source_fact_ids", []) if x in raw2id]
            for x in ids:
                self.coobs[x].update(i for i in ids if i != x)
        self.graph = h

        # retrieval text
        for F in self.facts.values():
            parts = [F["predicate"], F["value"] or ""] + [
                f"{r['role']} {r['desc']}" for r in F["roles"]
            ]
            F["text"] = " ; ".join(p for p in parts if p)
        for E in self.ent.values():
            owned = [self.facts[f]["value"] or "" for f in E["texts"]]
            E["text"] = " ; ".join(
                [d[1] for d in E["descs"]] + [a for d in E["descs"] for a in d[3]] + owned
            )
        self.fact_ids = list(self.facts)
        self.ent_ids = list(self.ent)
        self.duration = max((F["t1"] or 0) for F in self.facts.values()) if self.facts else 0
        self.role_vocab = Counter(
            r["role"] for F in self.facts.values() if not F["dropped"] for r in F["roles"]
        )
        # embedding
        self.emb = None
        if embed:
            em = Embedder(cache_dir)
            self.emb_fact = em.embed([self.facts[f]["text"] for f in self.fact_ids])
            self.emb_ent = em.embed([self.ent[a]["text"][:1500] for a in self.ent_ids])
            self.emb = em

    def _win_of_inst(self, iid):
        return self.win_of.get(iid.rsplit(":", 1)[0])

    def qvec(self, text):
        return self.emb.embed([text])[0] if self.emb else None

    @staticmethod
    def _cut(text, n):
        text = str(text or "")
        if n <= 0 or len(text) <= n:
            return text if n > 0 else ""
        cut = text[:n].rsplit(" ", 1)[0]
        return cut + "…"

    def render_fact(self, F, score=None, max_desc=70, seen=None):
        """seen: (entity, window) pairs already described in this response; later lines print the alias only."""
        head = f"{F['id']} [{mmss(F['t0'])}{'' if F['t1'] == F['t0'] else '–' + mmss(F['t1'])}]{' W' + str(F['window']) if F['window'] else ''} {F['kind']}"
        if F["kind"] == "text" and F.get("carrier"):
            head += f"({F['carrier']})"
        body = f' "{F["predicate"]}"' + (
            f' = "{F["value"]}"' if F["value"] not in (None, "") else ""
        )
        merged = {}
        for r in F["roles"]:  # one entity in several roles: merged into one segment
            k = (r["entity"], r["desc"])
            merged.setdefault(k, []).append(r["role"])
        parts = []
        for (ent, desc), rs in merged.items():
            key = (ent, F["window"])
            show = desc and max_desc > 0 and (seen is None or key not in seen or ent is None)
            if seen is not None and ent:
                seen.add(key)
            parts.append(
                f"{','.join(rs)}={ent or '?'}" + (f' "{self._cut(desc, max_desc)}"' if show else "")
            )
        roles = " ; ".join(parts)
        fl = sorted(F.get("flags", ()))
        extra = f" | ⚠ {', '.join(fl)}" if fl else ""
        if F["dropped"]:
            extra += f" | reason={F.get('reason')}"
        sc = f" | score={score:.3f}" if score is not None else ""
        return f"{head}{body} | {roles}{extra}{sc}"

    def render_entity(self, a, score=None):
        E = self.ent[a]
        descs = Counter(d[1] for d in E["descs"])
        top = " / ".join(f'"{d[:70]}"' for d, _ in descs.most_common(3))
        texts = [self.facts[f]["value"] for f in E["texts"] if self.facts[f]["value"]][:4]
        same = sorted({o for o, v, why, _ in self.idlinks.get(a, []) if v == "same"})
        s = (
            f"{a} {E['kind']} seen {mmss(E['first'])}–{mmss(E['last'])} | {len(E['facts'])} facts, "
            f"{len(E['instances'])} local instance(s) in {len({d[0] for d in E['descs']})} window(s) | {top}"
        )
        if texts:
            s += f" | text on/near it: {texts}"
        if same:
            s += f" | model judged SAME as (not merged): {same[:8]}"
        if score is not None:
            s += f" | score={score:.3f}"
        return s

    PERIODS = {
        "beginning": (0, 0.1),
        "end": (0.9, 1),
        "first_half": (0, 0.5),
        "second_half": (0.5, 1),
        "first_third": (0, 1 / 3),
        "middle_third": (1 / 3, 2 / 3),
        "last_third": (2 / 3, 1),
        "first_quarter": (0, 0.25),
        "second_quarter": (0.25, 0.5),
        "third_quarter": (0.5, 0.75),
        "last_quarter": (0.75, 1),
    }

    def resolve_time(
        self, start=None, end=None, around=None, tolerance=None, period=None, segment=None
    ):
        """Exact (start/end), around a moment (around +/- tolerance seconds, default 15), or coarse (period, or
        segment "k/n": the k-th of n equal parts). Several are intersected. Returns (start, end, note)."""
        s, e, why = parse_time(start), parse_time(end), []
        D = self.duration or 0

        def clip(a, b, tag):
            nonlocal s, e
            s = a if s is None else max(s, a)
            e = b if e is None else min(e, b)
            why.append(tag)

        if around not in (None, ""):
            c, tol = parse_time(around), float(tolerance or 15)
            if c is not None:
                clip(max(0, c - tol), c + tol, f"around {mmss(c)} ±{tol:g}s")
        if period in self.PERIODS:
            a, b = self.PERIODS[period]
            clip(a * D, b * D, f"{period} of the {mmss(D)} video")
        if segment and re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", str(segment)):
            k, n = (int(x) for x in str(segment).split("/"))
            if 1 <= k <= n:
                clip((k - 1) / n * D, k / n * D, f"segment {k}/{n}")
        if (start or end) and (around or period or segment):
            why.append("intersected with start/end")
        return s, e, "; ".join(why)

    def expand_fact(self, fid, mode):
        """author_closure = the co-observation hyperedge plus text on the same local participant in the same window;
        window = all records of the same observation window; participants = records in that window sharing a participant."""
        if not mode or mode == "none":
            return []
        F = self.facts[fid]
        same_win = [
            G
            for G in self.facts.values()
            if G["obs"] is not None and G["obs"] == F["obs"] and G["id"] != fid and not G["dropped"]
        ]
        if mode == "window":
            out = same_win
        elif mode == "participants":
            insts = {r["inst"] for r in F["roles"]}
            out = [G for G in same_win if insts & {r["inst"] for r in G["roles"]}]
        else:
            insts = {r["inst"] for r in F["roles"]}
            out = [self.facts[g] for g in self.coobs.get(fid, ())]
            out += [
                G
                for G in same_win
                if G["kind"] in ("text", "state", "attribute")
                and insts & {r["inst"] for r in G["roles"]}
            ]
        seen, res = set(), []
        for G in sorted(out, key=lambda G: G["t0"] or 0):
            if G["id"] not in seen:
                seen.add(G["id"])
                res.append(G["id"])
        return res[:12]

    def anchors(self, ids, max_items=8):
        """Extract anchors (exact words, names, aliases) from a result set so the next call can filter precisely."""
        Fs = [self.facts[i] for i in ids if i in self.facts]
        if not Fs:
            return ""
        ents = Counter(r["entity"] for F in Fs for r in F["roles"] if r["entity"])
        preds = Counter(F["predicate"] for F in Fs if F["kind"] == "event")
        vals = Counter(str(F["value"]) for F in Fs if F["value"] not in (None, ""))
        wins = Counter(F["window"] for F in Fs if F["window"])
        roles = Counter(f"{r['role']}" for F in Fs for r in F["roles"])
        out = ["anchors in these results (reuse them for precise follow-up queries):"]
        if ents:
            out.append(
                "  entities: " + ", ".join(f"{a}×{n}" for a, n in ents.most_common(max_items))
            )
        if preds:
            out.append(
                "  predicates (exact wording): "
                + "; ".join(f'"{p}"' for p, _ in preds.most_common(max_items))
            )
        if vals:
            out.append(
                "  values / on-screen text: "
                + "; ".join(f'"{v[:40]}"' for v, _ in vals.most_common(max_items))
            )
        if roles:
            out.append("  role names: " + ", ".join(r for r, _ in roles.most_common(max_items)))
        if wins:
            ws = sorted(wins)
            out.append(
                "  windows: "
                + ", ".join(
                    f"W{w}({mmss((self.win_span.get(w) or [None])[0])})×{wins[w]}"
                    for w in ws[: max_items * 2]
                )
            )
        return "\n".join(out)

    def get_entity(self, entity, **_):
        a = str(entity).strip().upper()
        E = self.ent.get(a)
        if not E:
            return f"no entity {entity}"
        lines = [
            self.render_entity(a),
            "local descriptions (window, first seen, description, attributes):",
        ]
        for w, d, t, attrs in E["descs"][:25]:
            lines.append(f'  W{w} {mmss(t)} "{d}"' + (f" {attrs}" if attrs else ""))
        if len(E["descs"]) > 25:
            lines.append(f"  … {len(E['descs']) - 25} more")
        if E["texts"]:
            lines.append("text / attribute / state facts owned by it:")
            lines += ["  " + self.render_fact(self.facts[f], max_desc=0) for f in E["texts"][:20]]
        preds = Counter(
            self.facts[f]["predicate"] for f in E["facts"] if self.facts[f]["kind"] == "event"
        )
        lines.append(f"event predicates (count): {dict(preds.most_common(15))}")
        links = self.idlinks.get(a, [])
        if links:
            lines.append(
                "identity comparisons with other entities (model verdict | what the author's code did | model's reason):"
            )
            for o, v, why, basis in sorted(links, key=lambda x: (x[1] != "same", x[0]))[:20]:
                lines.append(f"  {o}: {v} | {why} | {basis[:160]}" + self._contradiction(v, basis))
        else:
            lines.append("no cross-window identity comparison touched this entity")
        return "\n".join(lines)

    def get_window(self, window=None, time=None, raw_response=False, **_):
        """Raw records of one ~10 s observation window: local instances (description, attributes, boxes, entity), all facts, rejected records, optionally the raw VLM reply."""
        if window in (None, "") and time not in (None, ""):
            t = parse_time(time)
            window = next(
                (
                    w
                    for w, sp in sorted(self.win_span.items())
                    if sp[0] is not None and sp[0] <= t < (sp[1] or 1e9)
                ),
                None,
            )
        try:
            w = int(str(window).lstrip("Ww"))
        except (TypeError, ValueError):
            return f"no window {window}"
        oid = self.oid_of_win.get(w)
        if not oid:
            return f"no window {window}"
        o = self.obs_by_id[oid]
        sp = self.win_span.get(w) or [None, None]
        lines = [
            f"W{w} {mmss(sp[0])}–{mmss(sp[1])} structural_status={o.get('structural_status')}",
            "local instances declared by the vision model (local id → entity):",
        ]
        for i in o["observation"].get("instances", []):
            iid = i["instance_id"]
            a = self.alias.get(self.ent_of.get(iid))
            attrs = [
                f"{x.get('dimension')}: {x.get('value')}"
                for x in (i.get("attributes") or [])
                if isinstance(x, dict)
            ]
            fl = sorted(self.inst_flags.get(iid, ()))
            lines.append(
                f'  {iid.rsplit(":", 1)[-1]} → {a} {i["kind"]} "{i.get("description", "")}" {attrs if attrs else ""} boxes={len(i.get("regions", []))}'
                + (f" ⚠ {', '.join(fl)}" if fl else "")
            )
        recs = sorted(
            (G for G in self.facts.values() if G["obs"] == oid),
            key=lambda G: (G["dropped"], G["t0"] or 0),
        )
        lines.append(f"records ({len(recs)}):")
        lines += ["  " + self.render_fact(G, max_desc=0) for G in recs]
        other = [
            r for r in o.get("rejected_records", []) if not (r.get("record") or {}).get("predicate")
        ]
        if other:
            lines.append(f"other dropped records ({len(other)}), mostly malformed links:")
            lines += [
                f"  {r.get('reason')}: {json.dumps(r.get('record'), ensure_ascii=False)[:200]}"
                for r in other[:12]
            ]
        if raw_response:
            try:
                resp = _load(
                    self.run / "local_calls" / str(o.get("request_key")) / "response.json", {}
                )
                lines.append(
                    "raw vision-model response:\n" + resp["choices"][0]["message"]["content"]
                )
            except Exception:  # noqa: BLE001
                lines.append("(raw response not available)")
        return "\n".join(lines)

    @staticmethod
    def _contradiction(verdict, basis):
        """Flag identity verdicts whose free-text reason contradicts the verdict field."""
        b = str(basis or "").casefold()
        says_same = re.search(
            r"\b(is|are) the same\b|\bsame (person|woman|man|individual|girl|boy|child)\b", b
        ) and not re.search(r"\bnot the same\b", b)
        says_diff = re.search(
            r"\bdifferent (person|people|individuals?|woman|man)\b|\bnot the same\b", b
        )
        if verdict == "different" and says_same and not says_diff:
            return "  ⚠ reason text says SAME but verdict is different"
        if verdict == "same" and says_diff and not says_same:
            return "  ⚠ reason text says DIFFERENT but verdict is same"
        return ""

    def compare_entities(self, a, b, **_):
        """Are two aliases the same person/object? Side-by-side descriptions, direct comparisons, and co-occurrence in one window (usually two different people)."""
        A, Bn = str(a).strip().upper(), str(b).strip().upper()
        if A not in self.ent or Bn not in self.ent:
            return "unknown entity"
        wa = {d[0] for d in self.ent[A]["descs"]}
        wb = {d[0] for d in self.ent[Bn]["descs"]}
        both = sorted(wa & wb)
        direct = [x for x in self.idlinks.get(A, []) if x[0] == Bn]
        lines = [
            self.render_entity(A),
            self.render_entity(Bn),
            f"windows where BOTH were declared at the same time: {['W%d' % w for w in both] or 'none'}"
            + (
                " → usually two different individuals (unless the model duplicated one person in that window)"
                if both
                else ""
            ),
            "direct model comparisons: "
            + (
                "; ".join(
                    f"{v} | {why} | {basis[:200]}{self._contradiction(v, basis)}"
                    for _, v, why, basis in direct
                )
                or "none"
            ),
        ]
        ta = [d[1] for d in self.ent[A]["descs"]][:6]
        tb = [d[1] for d in self.ent[Bn]["descs"]][:6]
        lines.append(f"{A} descriptions: {ta}")
        lines.append(f"{Bn} descriptions: {tb}")
        if self.emb is not None:
            ia, ib = self.ent_ids.index(A), self.ent_ids.index(Bn)
            lines.append(
                f"description embedding similarity: {float(self.emb_ent[ia] @ self.emb_ent[ib]):.3f} (baseline for unrelated descriptions is ~0.5–0.6)"
            )
        return "\n".join(lines)


_ID_RE = re.compile(r"^[PORVporv]\d+$")


_CJK = re.compile("[\u4e00-\u9fff]")


def _terms(spec):
    """'blond|blonde "sit on" hoodie' -> [['blond','blonde'], ['sit on'], ['hoodie']]; quoted text is a phrase."""
    out = []
    for m in re.finditer(r'"([^"]+)"|(\S+)', str(spec)):
        tok = m.group(1) if m.group(1) is not None else m.group(2)
        alts = [a.strip() for a in tok.split("|") if a.strip()]
        if alts:
            out.append(alts)
    return out


def kw_match(text, spec):
    """Exact keywords: every term must occur (| = or inside a term). Latin text matches word prefixes, case-insensitive (jump hits jumps/jumping, not bungeejump); CJK text matches substrings."""
    text = str(text or "")
    for alts in _terms(spec):
        ok = False
        for a in alts:
            if _CJK.search(a):
                ok = a in text
            else:
                pat = r"(?<![A-Za-z0-9])" + r"\s+".join(re.escape(w) for w in a.split())
                ok = re.search(pat, text, re.I) is not None
            if ok:
                break
        if not ok:
            return False
    return True


_V5_PARAMS = {
    "query",
    "kinds",
    "predicate",
    "value",
    "roles",
    "ids",
    "window",
    "start",
    "end",
    "around",
    "period",
    "sort",
    "limit",
    "offset",
    "view",
    "include_dropped",
}


def _search5(
    self,
    query=None,
    kinds=None,
    predicate=None,
    value=None,
    roles=None,
    ids=None,
    window=None,
    start=None,
    end=None,
    around=None,
    period=None,
    sort=None,
    limit=10,
    offset=0,
    view="list",
    include_dropped=False,
    **extra,
):
    # unknown arguments / invalid values raise instead of being silently ignored
    if extra:
        return (
            f"error: unknown parameter(s) {sorted(extra)} — nothing was searched. Valid parameters: {sorted(_V5_PARAMS)}. "
            "Time uses start/end ('mm:ss'), around ('mm:ss±sec'), period, or window."
        )
    bad = []
    if view not in (None, "list", "window", "closure", "entity"):
        bad.append(
            f"view={view!r} (use list | window | closure | entity; put window numbers in `window`)"
        )
    if sort not in (None, "relevance", "time_asc", "time_desc"):
        bad.append(f"sort={sort!r} (use relevance | time_asc | time_desc)")
    ok_kinds = {"event", "text", "state", "attribute", "speech"}
    if kinds and (not isinstance(kinds, list) or not set(kinds) <= ok_kinds):
        bad.append(f"kinds={kinds!r} (a list from {sorted(ok_kinds)})")
    if roles is not None and not isinstance(roles, dict):
        bad.append("roles must be an object {slot: filler}")
    if bad:
        return "error: invalid value(s) — nothing was searched: " + "; ".join(bad)
    view = view or "list"
    ids = [ids] if isinstance(ids, str) else ids
    limit = max(1, min(int(limit or 10), 100))
    # ---- time
    tol = None
    if around not in (None, ""):
        m = re.match(r"\s*([\d:.]+)\s*(?:±|\+-|\+/-)\s*([\d.]+)\s*$", str(around))
        if m:
            around, tol = m.group(1), float(m.group(2))
    seg = period if period and "/" in str(period) else None
    s_, e_, twhy = self.resolve_time(start, end, around, tol, None if seg else period, seg)
    wins = window if isinstance(window, list) else ([window] if window not in (None, "") else [])
    wins = [re.sub(r"[^0-9]", "", str(w)) for w in wins]  # accept "W17" / '"W66"' / 17
    want_win = {int(w) for w in wins if w} or None
    notes = (
        [
            f"time filter: {mmss(s_) if s_ is not None else 'start'}–{mmss(e_) if e_ is not None else 'end'} ({twhy})"
        ]
        if twhy
        else []
    )

    # ---- view=entity: entity profile / pairwise comparison
    if view == "entity":
        ents = [str(i).strip().upper() for i in (ids or []) if _ID_RE.match(str(i).strip())]
        if not ents:
            return "view=entity needs entity ids in `ids`, e.g. ['P3'] or ['P3','P9'] (two ids → same-person check)"
        out = [self.get_entity(a) for a in ents]
        if len(ents) == 2:
            out.append("same-person check:\n" + self.compare_entities(ents[0], ents[1]))
        return "\n\n".join(out)
    # ---- window only (no other filter)
    if view == "window" and want_win and not any([query, kinds, predicate, value, roles, ids]):
        return "\n\n".join(self.get_window(window=w) for w in sorted(want_win))

    # ---- exact filters
    cons = []
    for slot, filler in (roles or {}).items():
        if filler in ("?", "*") or filler is None:
            cons.append((slot, "any", None))
        elif isinstance(filler, list):
            cons.append((slot, "ids", {str(x).strip().upper() for x in filler}))
        elif _ID_RE.match(str(filler).strip()):
            cons.append((slot, "ids", {str(filler).strip().upper()}))
        else:
            cons.append((slot, "kw", str(filler)))

    def slot_ok(F):
        for slot, mode, arg in cons:
            if slot == "*":
                cand = F["roles"]
            elif slot.startswith("@"):
                cand = [r for r in F["roles"] if r["group"] == slot]
            else:
                cand = [r for r in F["roles"] if r["role"].casefold() == slot.casefold()]
            if not cand:
                return False
            if mode == "ids" and not any(r["entity"] in arg for r in cand):
                return False
            if mode == "kw" and not any(kw_match(r["desc"], arg) for r in cand):
                return False
        return True

    want_ids = {str(i).strip().upper() for i in ids} if ids else None
    kinds = set(kinds or []) or None
    pool, n_drop = [], 0
    for fid in self.fact_ids:
        F = self.facts[fid]
        if want_ids is not None and fid not in want_ids:
            continue
        if kinds and F["kind"] not in kinds:
            continue
        if want_win is not None and F["window"] not in want_win:
            continue
        if s_ is not None and (F["t1"] is None or F["t1"] < s_):
            continue
        if e_ is not None and (F["t0"] is None or F["t0"] > e_):
            continue
        if predicate and not kw_match(F["predicate"], predicate):
            continue
        if value and not kw_match(F["value"], value):
            continue
        if cons and not slot_ok(F):
            continue
        if F["dropped"] and not include_dropped and want_ids is None:
            n_drop += 1
            continue
        pool.append(fid)

    # ---- de-duplication: same window, predicate and participants -> keep one
    groups = {}
    for fid in pool:
        F = self.facts[fid]
        k = (
            F["window"],
            norm_phrase(F["predicate"]),
            norm_phrase(F["value"]),
            tuple(sorted((r["role"], r["entity"]) for r in F["roles"])),
        )
        groups.setdefault(k, []).append(fid)
    dup_of = {}
    kept = []
    for k, fids in groups.items():
        kept.append(fids[0])
        dup_of[fids[0]] = len(fids)
    n_dup = len(pool) - len(kept)
    pool = kept

    # ---- ranking by sentence embedding; drop results more than 0.15 below the best similarity
    sim, n_low = {}, 0
    if query and self.emb is not None and pool:
        idx = {f: i for i, f in enumerate(self.fact_ids)}
        v = self.qvec(query)
        sim = {f: float(self.emb_fact[idx[f]] @ v) for f in pool}
        top = max(sim.values())
        keep = [f for f in pool if sim[f] >= top - 0.15]
        n_low = len(pool) - len(keep)
        pool = keep
    sort = sort or ("relevance" if query else "time_asc")
    if sort == "relevance" and sim:
        pool.sort(key=lambda f: -sim[f])
    elif sort == "time_desc":
        pool.sort(key=lambda f: -(self.facts[f]["t0"] or 0))
    else:
        pool.sort(
            key=lambda f: (self.facts[f]["t0"] if self.facts[f]["t0"] is not None else 1e9, f)
        )

    # ---- empty-window notice
    if s_ is not None or e_ is not None or want_win is not None:
        lo, hi = (s_ if s_ is not None else 0), (e_ if e_ is not None else 1e9)
        empty = [
            w
            for w, sp in sorted(self.win_span.items())
            if sp[0] is not None
            and sp[1] is not None
            and sp[1] > lo
            and sp[0] < hi
            and (want_win is None or w in want_win)
            and not any(G["window"] == w and not G["dropped"] for G in self.facts.values())
        ]
        if empty:
            notes.append(
                "memory recorded NOTHING in these windows (not evidence of absence): "
                + ", ".join(
                    f"W{w}({mmss(self.win_span[w][0])}–{mmss(self.win_span[w][1])})"
                    for w in empty[:20]
                )
            )
    excl = []
    if n_drop:
        excl.append(f"{n_drop} validator-dropped (include_dropped=true to see)")
    if n_dup:
        excl.append(f"{n_dup} duplicates in the same window (shown as ×k)")
    if n_low:
        excl.append(f"{n_low} with similarity > 0.15 below the best")
    total = len(pool)
    page = pool[offset : offset + limit]
    head = (
        f"{total} facts; showing {offset + 1 if page else 0}–{offset + len(page)} (sort={sort})"
        + (f"; excluded: {'; '.join(excl)}" if excl else "")
    )

    seen, lines = set(), []
    if view == "window":  # all records of the hit windows; hits marked with ▶
        hitset = set(page)
        for w in sorted({self.facts[f]["window"] for f in page if self.facts[f]["window"]})[:6]:
            sp = self.win_span.get(w) or [None, None]
            lines.append(f"--- W{w} {mmss(sp[0])}–{mmss(sp[1])} (all records; ▶ = matched)")
            for G in sorted(
                (
                    G
                    for G in self.facts.values()
                    if G["window"] == w and (include_dropped or not G["dropped"])
                ),
                key=lambda G: G["t0"] or 0,
            ):
                lines.append(("▶ " if G["id"] in hitset else "  ") + self.render_fact(G, seen=seen))
    else:
        for f in page:
            line = self.render_fact(self.facts[f], seen=seen)
            if dup_of.get(f, 1) > 1:
                line += f" | ×{dup_of[f]}"
            if sim:
                line += f" | sim={sim[f]:.2f}"
            lines.append(line)
            if view == "closure":
                for g in self.expand_fact(f, "author_closure"):
                    lines.append("    ↳ " + self.render_fact(self.facts[g], seen=seen))
    anchor_ids = page + (
        [g for f in page for g in self.expand_fact(f, "author_closure")]
        if view == "closure"
        else []
    )
    return "\n".join(
        notes + [head] + (lines or ["(none)"]) + ([self.anchors(anchor_ids)] if page else [])
    )


Memory.search = _search5
