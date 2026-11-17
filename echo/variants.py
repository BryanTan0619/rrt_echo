"""Memory variants for the controlled study (RQ3), all built FORWARD from the same cached observations.

Every builder takes the observation bundle of one video (observations.json) and returns the memory
file the agent reads (hypergraph.json). No variant is obtained by deleting structure from the ECHO
hypergraph; a time-free or coarse-clock variant records time differently when the memory is written.

  full                     ECHO hypergraph
  full_30s                 ECHO hypergraph on a 30-s clock
  no_temporal              ECHO hypergraph with no time
  no_identity              ECHO hypergraph without cross-window identity merges
  no_identity_no_temporal  both of the above
  binary                   binary graph: one node per entity; events split into pairwise links;
                           attributes / states / text / descriptions as separate links; exact times
  binary_30s               binary graph on a 30-s clock
  binary_untimed           binary graph with no time
  triple / triple_untimed  LightRAG-style merged triples (with / without time)
  flat / flat_untimed      template captions, one passage per observation window (with / without time)
"""
import collections
import copy
import hashlib
import json
import math
import re

from .rrt import ledger
from .rrt import representations as rep
from .schema import digest

VARIANTS = ('full', 'full_30s', 'no_temporal', 'no_identity', 'no_identity_no_temporal',
            'binary', 'binary_30s', 'binary_untimed', 'triple', 'triple_untimed', 'flat', 'flat_untimed')
SEGMENT = 30.0
AGENT_SPEECH_KEYS = ('utterance_id', 'predicate', 'speaker_id', 'text', 'start', 'end', 'speaker_status')


def _json(x):
    return json.loads(json.dumps(x))


# ---------------------------------------------------------------- hypergraph family
def hypergraph(bundle, identity=True):
    g, _ = rep.build_graph(bundle, 'hypergraph' if identity else 'no_identity')
    return _json(g)


def remove_time(g):
    """No clock: no observed times or extents; speech without start/end, in a time-free order."""
    for f in g['facts']:
        f['observed_times'] = []
        for k in ('extent', 'time_bounds'):
            f.pop(k, None)
    for i in g['instances']:
        i.pop('extent', None)
    a = g.get('audio_memory') or {}
    if a.get('utterances'):
        for u in a['utterances']:
            for k in ('start', 'end'):
                u.pop(k, None)
        a['utterances'].sort(key=lambda u: hashlib.sha256(json.dumps(u, sort_keys=True).encode()).hexdigest())
    return g


def coarse_clock(g, video, W=SEGMENT):
    """W-second clock: point times -> segment bounds, intervals -> covering segments; speech ids re-hashed."""
    lo = lambda t: math.floor(t / W) * W
    hi = lambda t: math.floor(t / W) * W + W
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)

    def walk(x):
        if isinstance(x, list):
            return [walk(v) for v in x]
        if not isinstance(x, dict):
            return x
        out = {}
        for k, v in x.items():
            if k == 'observed_times' and isinstance(v, list) and all(num(t) for t in v):
                out[k] = sorted({y for t in v for y in (lo(t), hi(t))})
            elif k in ('extent', 'interval', 'time_bounds') and isinstance(v, list) and len(v) == 2 and all(num(t) for t in v):
                out[k] = [lo(v[0]), hi(v[1])]
            elif k == 'observed_segments' and isinstance(v, list):
                out[k] = [[lo(s[0]), hi(s[1])] if isinstance(s, list) and len(s) == 2 and all(num(t) for t in s) else walk(s) for s in v]
            elif k in ('start', 'seconds') and num(v):
                out[k] = lo(v)
            elif k == 'end' and num(v):
                out[k] = hi(v)
            else:
                out[k] = walk(v)
        if 'pts' in out and num(out.get('seconds')) and isinstance(out.get('time_base'), list) and len(out['time_base']) == 2:
            a, b = out['time_base']
            out['pts'] = int(round(out['seconds'] * b / a))
        return out

    g = walk(g)
    s = json.dumps(g)
    pat = re.compile(re.escape(video) + r':speech:(\d+)(?!\d)')
    text = {u.get('utterance_id'): u.get('text') for u in (g.get('audio_memory') or {}).get('utterances', [])}
    ids = {old: f"{video}:speech:{digest([text.get(old), old])[:12]}" for old in sorted({m.group(0) for m in pat.finditer(s)})}
    g = json.loads(pat.sub(lambda m: ids[m.group(0)], s))
    a = g.get('audio_memory') or {}
    if a.get('utterances'):
        a['utterances'].sort(key=lambda u: (u.get('start') or 0, u.get('utterance_id')))
    return g


# ---------------------------------------------------------------- binary graph
def _segment(times, clock):
    if clock == 'none':
        return []
    if not clock:
        return list(times)
    return sorted({x for t in times for x in (math.floor(t / clock) * clock, math.floor(t / clock) * clock + clock)})


def binary_graph(bundle, speech, video, clock=None):
    """clock: None = exact seconds, a number = segment length in seconds, 'none' = no time."""
    g = ledger.primitive_source(bundle)
    ent_of = {li: e['entity_id'] for e in g['entities'] for li in e['local_instances']}
    inst = {i['instance_id']: i for i in g['instances']}
    node = lambda li: 'node:' + ent_of.get(li, li)
    facts = [f for f in g['facts'] if not f.get('retrieval_quarantined')]
    times_of = collections.defaultdict(list)
    for f in facts:
        for li in f['roles'].values():
            if li:
                times_of[li].extend(f.get('observed_times') or [])
    out, seen = [], set()

    def edge(kind, predicate, roles, times, value=None, carrier=None, unresolved=()):
        rec = {'kind': kind, 'predicate': predicate, 'value': value, 'carrier': carrier, 'roles': roles,
               'resolved_roles': {r: {'entity_id': n[len('node:'):]} for r, n in roles.items()},
               'observed_times': sorted(set(_segment(times, clock))), 'unresolved_slots': sorted(unresolved)}
        key = digest(rec)
        if key not in seen:
            seen.add(key)
            out.append({**rec, 'fact_id': 'l1:' + key[:24]})

    for f in facts:
        roles = {r: li for r, li in f['roles'].items() if li}
        if f['kind'] == 'event' and roles:
            for pair in rep.all_pairs(roles):
                edge('event', f['predicate'], {r: node(roles[r]) for r in pair}, f.get('observed_times') or [],
                     f.get('value'), f.get('carrier'), [s for s in f.get('unresolved_slots', []) if s in pair])
        elif roles:
            edge(f['kind'], f['predicate'], {r: node(li) for r, li in roles.items()}, f.get('observed_times') or [],
                 f.get('value'), f.get('carrier'), f.get('unresolved_slots', []))
    for li, i in inst.items():
        t = times_of.get(li, [])
        if i.get('description'):
            edge('attribute', 'description', {'owner': node(li)}, t, i['description'])
        for a in i.get('attributes') or []:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                edge('attribute', a[0], {'owner': node(li)}, t, a[1])

    kinds = collections.defaultdict(collections.Counter)
    span = collections.defaultdict(list)
    for li, i in inst.items():
        kinds[node(li)][i.get('kind') or 'object'] += 1
    for f in out:
        for n in f['roles'].values():
            span[n].extend(f['observed_times'])
    nodes = sorted(kinds)
    instances = [{'instance_id': n, 'kind': kinds[n].most_common(1)[0][0], 'description': '', 'attributes': [], 'regions': [],
                  'extent': [min(span[n]), max(span[n])] if span[n] else None} for n in nodes]
    entities = [{'entity_id': n[len('node:'):], 'local_instances': [n], 'status': 'linked'} for n in nodes]
    audio = {k: speech[k] for k in ('schema', 'source_video_id', 'rules') if k in speech}
    audio['utterances'] = [{k: u[k] for k in AGENT_SPEECH_KEYS if k in u} for u in speech.get('utterances', [])]
    if clock == 'none':
        for u in audio['utterances']:
            u.pop('start', None)
            u.pop('end', None)
            u['utterance_id'] = f"{video}:speech:{digest([u.get('text'), u['utterance_id']])[:12]}"
        audio['utterances'].sort(key=lambda u: u['utterance_id'])
    elif clock:
        for u in audio['utterances']:
            if u.get('start') is not None:
                u['start'] = math.floor(u['start'] / clock) * clock
            if u.get('end') is not None:
                u['end'] = math.floor(u['end'] / clock) * clock + clock
            u['utterance_id'] = f"{video}:speech:{digest([u.get('text'), u.get('start'), u.get('end'), u['utterance_id']])[:12]}"
        audio['utterances'].sort(key=lambda u: (u.get('start') or 0, u['utterance_id']))
    return {'schema': 'echo.binary_graph.v1', 'source_video_id': video, 'instances': instances, 'entities': entities,
            'facts': sorted(out, key=lambda f: f['fact_id']), 'event_hyperedges': [], 'binding_links': [],
            'endpoint_grounding_gaps': [], 'audio_memory': audio}


# ---------------------------------------------------------------- text memories (triple / flat)
def _untimed_primitive(bundle):
    g = _primitive(bundle)
    for f in g['facts']:
        f['observed_times'] = []
    for i in g['instances']:
        i.pop('extent', None)
    return g


_primitive = ledger.primitive_source


def _untimed_utterances(bundle, video):
    us = []
    for u in bundle.get('audio_memory', {}).get('utterances', []):
        u = {k: v for k, v in u.items() if k not in ('start', 'end', 'words', 'segments')}
        u['utterance_id'] = f"{video}:speech:{digest([u.get('text'), u.get('utterance_id')])[:12]}"
        us.append(u)
    return sorted(us, key=lambda u: u['utterance_id'])


def triple_documents(bundle, video, timed=True):
    if timed:
        return rep.triple_documents(bundle, 500)[0]
    b = copy.deepcopy(bundle)
    b['audio_memory'] = {**b.get('audio_memory', {}), 'utterances': _untimed_utterances(bundle, video)}
    ledger.primitive_source = _untimed_primitive
    try:
        return rep.triple_documents(b, 500)[0]
    finally:
        ledger.primitive_source = _primitive


def flat_documents(bundle, video, timed=True, speech_seconds=60.0):
    """Template captions (one passage per observation window) plus speech passages.
    timed=True reproduces rep.nl_documents exactly; timed=False drops every time prefix, orders sentences
    and passages by content, and packs speech in a time-free order into the same number of passages."""
    g = ledger.primitive_source(bundle)
    inst = {i['instance_id']: i for i in g['instances']}

    def who(x):
        d = (inst.get(x, {}).get('description') or inst.get(x, {}).get('kind') or 'someone').strip().rstrip('.')
        return d[:1].lower() + d[1:] if d[:2] != d[:2].upper() else d

    windows = {}
    for f in g['facts']:
        if f.get('retrieval_quarantined'):
            continue
        roles = {r: x for r, x in (f.get('roles') or {}).items() if x}
        times = f.get('observed_times') or []
        key = f['fact_id'].rsplit(':', 2)[0] if f['fact_id'].count(':') >= 4 else 'misc'
        if f['kind'] == 'event':
            parts = '; '.join(f'{r.replace("_", " ")}: {who(x)}' for r, x in sorted(roles.items()))
            body = f"{f['predicate']}" + (f' ({parts})' if parts else '') + '.'
        elif f['kind'] == 'text':
            owner = next(iter(roles.values()), None)
            body = f"the text \"{f.get('value')}\" is shown" + (f' on {who(owner)}' if owner else '') + '.'
        else:
            owner = next(iter(roles.values()), None)
            value = f.get('value') if f.get('value') is not None else f.get('carrier')
            body = f"{who(owner) if owner else 'the scene'} — {f['predicate']}: {value}."
        text = (rep._span(times).capitalize() + ', ' + body) if timed else body[:1].upper() + body[1:]
        w = windows.setdefault(key, {'sentences': [], 'times': []})
        w['sentences'].append(((min(times) if times else 0.0) if timed else text, text))
        w['times'] += list(times)
    if timed:
        order = sorted(windows.items(), key=lambda kv: (min(kv[1]['times']) if kv[1]['times'] else 0.0, kv[0]))
    else:
        order = sorted(windows.items(), key=lambda kv: digest(sorted(t for _, t in kv[1]['sentences'])))
    docs = []
    for _, w in order:
        text = ' '.join(t for _, t in sorted(w['sentences']))
        d = {'id': ('N' + str(len(docs))) if timed else 'N' + digest(text)[:12], 'text': text}
        if timed:
            d['times'] = sorted(set(w['times']))
        docs.append(d)
    utts = sorted(bundle.get('audio_memory', {}).get('utterances', []), key=lambda u: u.get('start', 0))
    blocks, block, start = [], [], None
    for u in utts:
        if start is not None and u.get('start', 0) - start > speech_seconds:
            blocks.append(block)
            block, start = [], None
        if start is None:
            start = u.get('start', 0)
        block.append(u)
    if block:
        blocks.append(block)
    if timed:
        for b in blocks:
            docs.append({'id': 'N' + str(len(docs)),
                         'text': ' '.join(f'From {u["start"]:.1f}s to {u["end"]:.1f}s, someone says: "{u.get("text", "").strip()}".' for u in b),
                         'speech': [[u['start'], u['end']] for u in b]})
    elif utts:
        hu = _untimed_utterances(bundle, video)
        size = -(-len(hu) // len(blocks))
        for i in range(0, len(hu), size):
            text = ' '.join(f'Someone says: "{u.get("text", "").strip()}".' for u in hu[i:i + size])
            docs.append({'id': 'N' + digest(text)[:12], 'text': text})
    return docs


def text_memory(name, docs):
    """Passages as `text` records: the agent searches them like any other record."""
    facts = [{'fact_id': name + ':' + x['id'], 'kind': 'text', 'predicate': 'passage', 'value': x['text'],
              'carrier': name + '_text', 'roles': {}, 'resolved_roles': {}, 'observed_times': sorted(x.get('times') or []),
              'unresolved_slots': []} for x in docs]
    return {'instances': [], 'entities': [], 'facts': facts, 'endpoint_grounding_gaps': [], 'binding_links': [],
            'event_hyperedges': [], 'audio_memory': {}}


# ---------------------------------------------------------------- entry point
def build(bundle, variants=VARIANTS):
    """Return {variant: memory} for one video."""
    video = bundle['video_id']
    out = {}
    full = hypergraph(bundle)
    need = set(variants)
    if 'full' in need:
        out['full'] = full
    if 'full_30s' in need:
        out['full_30s'] = coarse_clock(_json(full), video)
    if 'no_temporal' in need:
        out['no_temporal'] = remove_time(_json(full))
    if need & {'no_identity', 'no_identity_no_temporal'}:
        noid = hypergraph(bundle, identity=False)
        if 'no_identity' in need:
            out['no_identity'] = noid
        if 'no_identity_no_temporal' in need:
            out['no_identity_no_temporal'] = remove_time(_json(noid))
    speech = full.get('audio_memory') or {}
    for name, clock in (('binary', None), ('binary_30s', SEGMENT), ('binary_untimed', 'none')):
        if name in need:
            out[name] = binary_graph(bundle, speech, video, clock)
    for name, timed in (('triple', True), ('triple_untimed', False)):
        if name in need:
            out[name] = text_memory(name, triple_documents(bundle, video, timed))
    for name, timed in (('flat', True), ('flat_untimed', False)):
        if name in need:
            out[name] = text_memory(name, flat_documents(bundle, video, timed))
    return out
