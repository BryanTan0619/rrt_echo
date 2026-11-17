"""Compact record form of an utterance (used when speech is packed into text documents)."""
from __future__ import annotations

# Binding content of a fact: predicate, role fillers, value, time, open slots.
# Each fact is delivered once; identity lives on the instance (entity_id), not
# repeated per role; frame-id lists carry nothing a text reader can use.
FACT_FIELDS = ('fact_id', 'kind', 'predicate', 'roles', 'value', 'observed_times',
               'unresolved_slots', 'carrier', 'retrieval_quarantined', 'event_node_id')


def seconds(media):
    pts, base = media.get('pts'), media.get('time_base')
    if pts is None or not base:return media.get('seconds')
    return round(pts * base[0] / base[1], 3)


# Derived restatements of facts already delivered, static rule text, and
# relations that assert nothing ("unresolved"). audio_nodes restate utterances.
UTTERANCE_FIELDS = ('utterance_id', 'text', 'start', 'end', 'speaker', 'speaker_id')


def compact_utterance(u):
    """One record per utterance: what was said and when; ASR diagnostics dropped."""
    return {k: u[k] for k in UTTERANCE_FIELDS if u.get(k) not in (None, '', [], {})}
