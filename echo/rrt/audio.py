"""Timestamped ASR and anonymous voice hypotheses, independent of visual identity.

Uses the local M3-Agent Whisper/ERes2NetV2 checkpoints and its Apache-2.0
speakerlab implementation. ASR segments are NOT guaranteed speaker turns.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

from ..schema import digest
from ..storage import atomic_json


def validate_audio(audio):
    if audio.get("artifact_id") != digest({k: v for k, v in audio.items() if k != "artifact_id"}):
        raise ValueError("audio_digest_mismatch")
    speakers = {s["speaker_id"] for s in audio["speakers"]}
    if len(speakers) != len(audio["speakers"]):
        raise ValueError("duplicate_audio_speaker")
    ids = set()
    for u in audio["utterances"]:
        if u["utterance_id"] in ids:
            raise ValueError("duplicate_utterance")
        ids.add(u["utterance_id"])
        if not all(math.isfinite(u[k]) for k in ("start", "end")) or u["end"] < u["start"]:
            raise ValueError("invalid_utterance_time")
        if u["speaker_id"] is not None and u["speaker_id"] not in speakers:
            raise ValueError("dangling_audio_speaker")
        if any(u["evidence"][k] != u[k] for k in ("start", "end")):
            raise ValueError("utterance_evidence_time_mismatch")
    if audio["source"]["status"] == "no_audio_stream" and audio["utterances"]:
        raise ValueError("speech_without_audio")


def attach_audio(graph, audio):
    expected = graph.get("source_video_id") or graph["video_id"]
    if expected != audio["source_video_id"]:
        raise ValueError("audio_video_id_mismatch")
    validate_audio(audio)
    # Check source identity against visual evidence when source hashes are available.
    hashes = {m.get("source_sha256") for m in graph.get("evidence", {}).values()} - {None}
    if hashes and audio["source"]["sha256"] not in hashes:
        raise ValueError("audio_video_hash_mismatch")
    out = copy.deepcopy(graph)
    out["audio_memory"] = copy.deepcopy(audio)
    # The graph's own on-screen text decides which single words can be names.
    out["audio_memory"]["on_screen_names"] = on_screen_name_vocabulary(out)
    out["audio_memory"]["mentions"] = extract_named_mentions(
        out["audio_memory"], on_screen_names=out["audio_memory"]["on_screen_names"])
    out["audio_memory"]["artifact_id"] = digest(
        {k: v for k, v in out["audio_memory"].items() if k != "artifact_id"}
    )
    out = link_audiovisual(out)
    out.pop("snapshot_id", None)
    out["snapshot_id"] = digest(out)
    return out


NAME_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")
OVERLAPPING_NAME_RE = re.compile(r"(?=(\b[A-Z][a-z]+\s+[A-Z][a-z]+\b))")
COLOR_RE = re.compile(r"\b(yellow|blue|red|green|white|black|orange|purple)\b")
NUMBER_RE = re.compile(r"number (\d+)")
# Common narration/league phrases that are not player referents.
NAME_STOP = {
    "eastern conference", "western conference", "new york", "indiana pacers",
    "nba com", "top five", "final four", "los angeles",
}


# Broadcast furniture a scoreboard OCR reads alongside the competitors.
ON_SCREEN_STOP = {
    "MATCH", "MATCHES", "POINT", "POINTS", "DEUCE", "GAME", "GAMES", "SET", "SETS",
    "ADVANTAGE", "SERVE", "SERVING", "BREAK", "TIEBREAK", "FINAL", "SEMIFINAL",
    "QUARTERFINAL", "ROUND", "COURT", "LIVE", "REPLAY", "HIGHLIGHTS", "OPEN",
    "SCORE", "SCORES", "TOTAL", "STATS", "WINNER", "ERROR", "ERRORS", "ACES",
    "FAULT", "CHALLENGE", "REVIEW", "SECONDS", "MINUTES", "HOURS",
}


def on_screen_name_vocabulary(graph):
    """Recall names only from name-shaped video text, never arbitrary OCR words.

    A score row with separate small numbers and a mixed-case full name are two
    visual naming layouts. Ingredient labels and chapter titles remain text facts,
    but do not turn every spoken ingredient into a person-name candidate. This
    only retrieves candidates; the audiovisual reviewer must still bind the name.
    """
    words = set()
    labels = {"Chef", "Dr", "Doctor", "Mr", "Mrs", "Ms", "Professor",
              "Director", "Featuring", "Editor", "Producer", "Camera"}
    for fact in graph.get("facts", []):
        if fact.get("kind") != "text":
            continue
        value = str(fact.get("value", ""))
        for match in OVERLAPPING_NAME_RE.finditer(value):
            left, right = match.group(1).split()
            if left not in labels and right not in labels and (
                left.upper() not in ON_SCREEN_STOP and right.upper() not in ON_SCREEN_STOP
            ):
                words.update((left, right))
        # A scoreboard may OCR a number before the surname, and is sometimes
        # tagged as screen rather than overlay. Require two separate score cells.
        for match in re.finditer(r"\b[A-Z][A-Z'\-]{2,}\b", value):
            token = match.group(0)
            if token in ON_SCREEN_STOP:
                continue
            tail = value[match.end():]
            next_word = re.search(r"\b[A-Za-z][A-Za-z'\-]{2,}\b", tail)
            if next_word:
                tail = tail[:next_word.start()]
            if len(re.findall(r"(?<![:\d])\d{1,2}(?![:\d])", tail)) >= 2:
                words.add(token)
    return sorted(words)


def _phonetic_key(word):
    """Crude consonant skeleton: enough to pair a misheard spelling, no more."""
    text = re.sub(r"[^a-z]", "", word.casefold())
    text = re.sub(r"(.)\1+", r"\1", text)
    for a, b in (("ph", "f"), ("ck", "k"), ("kh", "k"), ("ch", "k"), ("sh", "s"),
                 ("th", "t"), ("wr", "r"), ("gh", "g"), ("qu", "k"), ("x", "ks"),
                 ("z", "s"), ("v", "f"), ("w", "v"), ("y", "i")):
        text = text.replace(a, b)
    return re.sub(r"[aeiou]", "", text) or text


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def on_screen_spelling(token, vocabulary):
    """How a spoken token corresponds to a name the video displayed, if at all.

    Three outcomes, and the caller is told which:
      exact     the same word, ignoring case. The scoreboard shouts MUCHOVA and
                the commentator says Muchova; only the casing differs.
      phonetic  the consonant skeletons agree closely. A recogniser that has
                never seen the surname writes what it heard -- "mukavat" for
                Muchova -- and that spelling appears in no caption and on no
                scoreboard. This is a candidate for review, never an accepted
                name: similarity proposes, evidence disposes.
      None      no correspondence; the token keeps whatever it is.
    """
    for word in vocabulary:
        if token.casefold() == word.casefold():
            return {"on_screen": word, "match": "exact", "distance": 0}
    best = None
    spoken = _phonetic_key(token)
    if len(spoken) >= 3:
        for word in vocabulary:
            key = _phonetic_key(word)
            if len(key) < 3:
                continue
            distance = _edit_distance(spoken, key)
            if distance <= max(1, min(len(spoken), len(key)) // 3):
                if best is None or distance < best["distance"]:
                    best = {"on_screen": word, "match": "phonetic", "distance": distance}
    return best


def _name_word_span(utterance, name, occurrence=0):
    """Use ASR word times when the literal name aligns; otherwise keep utterance time."""
    words = utterance.get("words") or []
    target = re.findall(r"[a-z0-9]+", name.casefold())
    matches = []
    if target:
        for i in range(len(words) - len(target) + 1):
            tokens = ["".join(re.findall(r"[a-z0-9]+", w.get("text", "").casefold()))
                      for w in words[i:i + len(target)]]
            if tokens == target and all("start" in w and "end" in w
                                        for w in words[i:i + len(target)]):
                matches.append((words[i]["start"], words[i + len(target) - 1]["end"]))
    return matches[occurrence] if occurrence < len(matches) else (
        utterance["start"], utterance["end"])


def extract_named_mentions(audio, *, on_screen_names=()):
    """Literal ASR name spans for query-blind referent review; no identity is inferred.

    Two patterns count as a name. A pair of capitalised words is the commentary
    form used where a full name is spoken. A single word counts only when this
    video displayed that name on screen, which is what a sport whose commentary
    says just the surname needs -- "Muchova", never "Karolina Muchova". Requiring
    the word to have been seen on screen is also what keeps the single-word form
    from swallowing every sentence-initial "The" or "Oh".

    `text` stays the literal span as transcribed, so the downstream check that a
    mention really occurs in its utterance still holds. When the spoken spelling
    differs from the on-screen one, the correspondence travels beside it as a
    proposal with its own status, never silently replacing what was heard.
    """
    rows = []
    vocabulary = [w for w in on_screen_names if w.casefold() not in NAME_STOP]
    for u in audio.get("utterances", []):
        n = 0
        occurrences = {}
        spans = {}
        for match in OVERLAPPING_NAME_RE.finditer(u.get("text", "")):
            spans[match.start(1)] = (match.group(1), None)
        if vocabulary:
            for match in re.finditer(r"\b[A-Za-z][A-Za-z'\-]{2,}\b", u.get("text", "")):
                if any(start <= match.start() < start + len(text)
                       for start, (text, _) in list(spans.items())):
                    continue  # already inside a two-word name
                correspondence = on_screen_spelling(match.group(0), vocabulary)
                if correspondence:
                    spans.setdefault(match.start(), (match.group(0), correspondence))
        for position in sorted(spans):
            text, correspondence = spans[position]
            if text.casefold() in NAME_STOP:
                continue
            key = text.casefold()
            start, end = _name_word_span(u, text, occurrences.get(key, 0))
            occurrences[key] = occurrences.get(key, 0) + 1
            row = {
                "mention_id": f"{u['utterance_id']}:m{n}",
                "utterance_id": u["utterance_id"],
                "text": text,
                "char_span": [position, position + len(text)],
                "start": start,
                "end": end,
                "source": "literal_asr_span_not_identity",
            }
            if correspondence:
                row["on_screen_spelling"] = {
                    **correspondence,
                    "status": ("same_word_different_case"
                               if correspondence["match"] == "exact"
                               else "proposed_from_sound_not_accepted"),
                }
            rows.append(row)
            n += 1
    return rows


def link_audiovisual(graph, gap_seconds=5.0):
    """Register name<->jersey candidate bindings from temporal co-occurrence.

    A named referent in the transcript (a player name the commentary mentions) is
    registered against temporally-proximal jersey-number attribute facts. A
    referent is accepted only when one exact jersey owner is the actor of one
    overlapping event and the utterance action agrees with that event predicate.
    Other temporal co-occurrences remain unresolved candidates. This never binds
    the named referent as the speaker.
    """
    audio = graph.get("audio_memory")
    if not audio or not audio.get("utterances"):
        return graph
    jerseys = []
    for f in graph.get("facts", []):
        if f["kind"] != "attribute":
            continue
        value = f.get("value") or ""
        number = NUMBER_RE.search(value)
        if not number:
            continue
        times = f.get("observed_times") or []
        if not times:
            continue
        color = COLOR_RE.search(value)
        owner_roles = f.get("resolved_roles", {}).get("owner") or {}
        jerseys.append(
            {
                "number": number.group(1),
                "color": color.group(1) if color else None,
                "instance_id": f["roles"].get("owner"),
                "entity_id": owner_roles.get("entity_id") if isinstance(owner_roles, dict) else None,
                "t0": min(times),
                "t1": max(times),
            }
        )
    if not jerseys:
        return graph
    # Join a jersey observation to an occurrence only through its exact local
    # endpoint and overlapping evidence time.  This supports named referents in
    # commentary without pretending that the named player is the speaker.
    events = [f for f in graph.get("facts", []) if f.get("kind") == "event"]
    for jersey in jerseys:
        jersey["event_fact_ids"] = [
            f["fact_id"] for f in events
            if jersey["instance_id"] in {
                f.get("roles", {}).get(role)
                for role in ("agent", "holder", "writer")
            }
            and any(jersey["t0"] - gap_seconds <= t <= jersey["t1"] + gap_seconds
                    for t in f.get("observed_times", []))
        ]
    # Collect full names and their single-word fragments first, so a commentary
    # that says only "Tyrese" still links to "Tyrese Haliburton".
    full_names = set()
    fragments = set()
    for u in audio["utterances"]:
        for match in NAME_RE.finditer(u["text"]):
            full = f"{match.group(1)} {match.group(2)}"
            if full.casefold() in NAME_STOP:
                continue
            full_names.add(full)
            fragments.add(match.group(1))
            fragments.add(match.group(2))

    def names_in(text):
        found = []
        for match in NAME_RE.finditer(text):
            full = f"{match.group(1)} {match.group(2)}"
            if full.casefold() not in NAME_STOP:
                found.append(full)
        for word in re.findall(r"\b[A-Z][a-z]+\b", text):
            if word in fragments and word not in {n.split()[-1] for n in found}:
                found.append(word)
        return found

    def action_terms(text):
        aliases = {
            "dunks": "dunk", "dunked": "dunk", "dunking": "dunk",
            "passes": "pass", "passed": "pass", "passing": "pass", "lob": "pass", "lobs": "pass",
            "shoots": "shoot", "shot": "shoot", "shooting": "shoot",
            "scores": "score", "scored": "score", "scoring": "score",
            "dribbles": "dribble", "dribbled": "dribble", "dribbling": "dribble",
            "catches": "catch", "caught": "catch", "catching": "catch",
            "drives": "drive", "drove": "drive", "driving": "drive",
        }
        tokens = {aliases.get(w, w) for w in re.findall(r"[a-z]+", text.casefold())}
        return tokens & {"dunk", "pass", "shoot", "score", "dribble", "catch", "drive"}

    event_by_id = {f["fact_id"]: f for f in events}
    links = []
    for u in audio["utterances"]:
        utterance_actions = action_terms(u["text"])
        for name in names_in(u["text"]):
            nearby = [
                j
                for j in jerseys
                if min(u["end"], j["t1"]) - max(u["start"], j["t0"]) >= -gap_seconds
            ]
            if not nearby:
                continue
            for candidate in nearby:
                candidate["aligned_event_fact_ids"] = [
                    fid for fid in candidate["event_fact_ids"]
                    if utterance_actions & action_terms(event_by_id[fid].get("predicate", ""))
                ]
            accepted = (
                len(nearby) == 1
                and len(nearby[0]["aligned_event_fact_ids"]) == 1
            )
            links.append(
                {
                    "link_id": f"av:{u['utterance_id']}:{name.casefold()}",
                    "relation": "named_referent_jersey",
                    "name": name,
                    "confidence": (
                        "joint_event_unique"
                        if accepted
                        else "provisional_unique" if len(nearby) == 1 else "ambiguous"
                    ),
                    "status": (
                        "accepted_referent"
                        if accepted
                        else "unresolved"
                    ),
                    "utterance_id": u["utterance_id"],
                    "candidates": [
                        {
                            "number": j["number"],
                            "color": j["color"],
                            "instance_id": j["instance_id"],
                            "entity_id": j["entity_id"],
                            "event_fact_ids": j["event_fact_ids"],
                            "aligned_event_fact_ids": j["aligned_event_fact_ids"],
                        }
                        for j in nearby
                    ],
                    "basis": "utterance_name_action_plus_exact_jersey_owner_plus_aligned_event",
                    "evidence": {
                        "utterance_start": u["start"],
                        "utterance_end": u["end"],
                        "proximity_seconds": gap_seconds,
                    },
                }
            )
    audio["audiovisual_links"] = links
    audio["artifact_id"] = digest({k: v for k, v in audio.items() if k != "artifact_id"})
    return graph
