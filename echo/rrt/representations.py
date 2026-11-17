"""Representations built from one observation bundle.

  build_graph(bundle, 'hypergraph')   ECHO: one role-labelled hyperedge per observed event
  build_graph(bundle, 'no_identity')  the same, without cross-window same-identity merges
  triple_documents(bundle)            LightRAG-style triples merged across occurrences
  all_pairs(roles)                    role pairs used to split an event into pairwise links
"""
import copy
import itertools

from ..schema import digest
from . import ledger


def all_pairs(roles):
    if not roles:raise ValueError('event_without_roles')
    names = sorted(roles)
    return [tuple(p) for p in itertools.combinations(names, 2)] or [(names[0],)]


def without_identity_merges(bundle):
    """Same observations; drop every cross-window same-identity proposal."""
    b = copy.deepcopy(bundle);dropped = set()
    for e in b['identity_journal']:
        p = e.get('proposal') or {}
        if p.get('relation') == 'same_identity':dropped.add(p.get('proposal_id'))
    keep = []
    for e in b['identity_journal']:
        p = e.get('proposal') or {}
        ref = p.get('proposal_id') or e.get('proposal_id')
        if ref in dropped:continue
        keep.append(e)
    b['identity_journal'] = keep
    return b, {'dropped_same_identity_proposals': len(dropped), 'kept_journal_entries': len(keep)}


def build_graph(bundle, arm):
    """ECHO hypergraph ('hypergraph') or the same without identity merges ('no_identity')."""
    if arm == 'hypergraph':
        return ledger.build_graph(bundle, 'hypergraph'), {}
    if arm == 'no_identity':
        stripped, meta = without_identity_merges(bundle)
        return ledger.build_graph(stripped, 'hypergraph'), meta
    raise ValueError('unknown graph: ' + arm)


# --------------------------------------------------------------------------
# LightRAG-style triple graph.
#
# Entities are the identity-projected global entities; their descriptions
# absorb attribute/state/text values WITHOUT time, as LightRAG folds them into
# entity descriptions. Each event becomes (head, predicate, tail) relations for
# every role pair (one-role events become (entity, predicate)); relations with
# the same endpoints and predicate MERGE across occurrences, keeping only the
# merged time list and an occurrence count. No event key survives, so which
# participants co-occurred in which occurrence is not recoverable. ASR enters
# as text chunks, as LightRAG/HyperGraphRAG fuse chunks with graph records.
# --------------------------------------------------------------------------
def _json(row):
    import json
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def triple_documents(bundle, chunk_words=500, relation_cap=8):
    g = ledger.primitive_source(bundle)
    entity_of = {}
    for e in g['entities']:
        for i in e.get('local_instances', []):entity_of[i] = e['entity_id']
    instances = {i['instance_id']: i for i in g['instances']}
    ent = lambda x: entity_of.get(x, x)
    entities = {}
    def entity(eid):
        return entities.setdefault(eid, {'instance_id': eid, 'kind': None, 'descriptions': [], 'attributes': set(), 'names': set()})
    for iid, inst in instances.items():
        e = entity(ent(iid));e['kind'] = e['kind'] or inst.get('kind')
        d = inst.get('description')
        if d and d not in e['descriptions'] and len(e['descriptions']) < 3:e['descriptions'].append(d)
    for b in g.get('name_bindings', []):
        target = b.get('person_instance') or b.get('instance')
        if target and b.get('name'):entity(ent(target))['names'].add(b['name'])
    relations = {}
    for f in g['facts']:
        if f.get('retrieval_quarantined'):continue
        roles = {r: ent(x) for r, x in f.get('roles', {}).items() if x}
        times = f.get('observed_times', [])
        if f['kind'] != 'event':
            for x in roles.values():
                value = f.get('value') if f.get('value') is not None else f.get('carrier')
                entity(x)['attributes'].add(f"{f['predicate']}: {value}")
            continue
        pairs = list(itertools.combinations(sorted(roles), 2)) or [(r,) for r in roles]
        for pair in pairs:
            ends = tuple(roles[r] for r in pair)
            key = (ends, f['predicate'], pair)
            rel = relations.setdefault(key, {'predicate': f['predicate'], 'roles': {r: roles[r] for r in pair},
                                             'observed_times': set(), 'occurrences': 0})
            rel['observed_times'].update(times);rel['occurrences'] += 1
            for x in ends:entity(x)
    def entity_line(eid):
        e = entities[eid]
        row = {'instance_id': eid, 'kind': e['kind'], 'description': ' | '.join(e['descriptions']),
               'attributes': sorted(e['attributes']), 'names': sorted(e['names'])}
        return 'Observed instance: ' + _json(row)
    rel_rows = {}
    for (ends, pred, pair), rel in relations.items():
        rid = 'rel:' + digest([ends, pred, pair])[:24]
        rel_rows[rid] = {'fact_id': rid, 'kind': 'relation', 'predicate': pred, 'roles': rel['roles'],
                         'observed_times': sorted(rel['observed_times']), 'occurrences': rel['occurrences']}
    by_entity = {}
    for rid, row in rel_rows.items():
        for x in set(row['roles'].values()):by_entity.setdefault(x, []).append(rid)
    fact_line = lambda rid: 'Observed fact: ' + _json(rel_rows[rid])
    docs = []
    for rid, row in sorted(rel_rows.items()):
        lines = [fact_line(rid)] + [entity_line(x) for x in sorted(set(row['roles'].values()))]
        docs.append({'id': rid, 'text': '\n'.join(lines), 'seed': None})
    for eid in sorted(entities):
        top = sorted(by_entity.get(eid, []), key=lambda r: (-rel_rows[r]['occurrences'], r))[:relation_cap]
        docs.append({'id': 'ent:' + digest(eid)[:24], 'text': '\n'.join([entity_line(eid)] + [fact_line(r) for r in top]), 'seed': None})
    pending, words = [], 0
    from .compact_view import compact_utterance
    for u in bundle['audio_memory']['utterances']:
        line = 'ASR utterance: ' + _json(compact_utterance(u));n = len(line.split())
        if pending and words + n > chunk_words:
            docs.append({'id': 'asr:' + str(len(docs)), 'text': '\n'.join(pending), 'seed': None});pending, words = [], 0
        pending.append(line);words += n
    if pending:docs.append({'id': 'asr:' + str(len(docs)), 'text': '\n'.join(pending), 'seed': None})
    return docs, {'entities': len(entities), 'relations': len(rel_rows),
                  'merged_occurrences': sum(r['occurrences'] for r in rel_rows.values()),
                  'events': sum(f['kind'] == 'event' for f in g['facts']), 'documents': len(docs),
                  'representation': 'lightrag_style_triples_merged_across_occurrences'}


# --------------------------------------------------------------------------
# Natural-language caption memory (text-only baseline).
#
# Each observation window becomes one caption passage written by a fixed
# template from the same primitive records. Participants appear only as their
# window-local descriptions: no instance/entity IDs, no cross-window identity,
# no name bindings, no identity decisions. Times are kept per sentence so the
# variant differs from ECHO in organization, not in temporal information.
# Speech is packed into time-ordered passages. Provenance (which fact and which
# instance each sentence came from) is written to a side file for diagnostics
# only and never delivered.
# --------------------------------------------------------------------------
def _span(times):
    if not times:return ''
    lo, hi = min(times), max(times)
    return f'at {lo:.1f}s' if hi - lo < 0.05 else f'from {lo:.1f}s to {hi:.1f}s'
