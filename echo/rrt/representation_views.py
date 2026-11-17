"""Shared ID neutralisation applied to every graph before it is stored."""
from collections import Counter
from ..schema import digest
from ..scoped_identity import identity_paths
from .hypergraph import TemporalEvidenceHypergraph, seal
from .event_content import attach_coobserved_attributes


def neutralize_ids(graph):
    """Replace occurrence-specific ID prefixes with neutral content hashes."""
    prefixes=('video:','entity:','identity:','association:','coobserved:','name-binding:','name-mention:')
    mapping={}
    def collect(x):
        if isinstance(x,dict):
            for k,v in x.items():collect(k);collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
        elif isinstance(x,str) and x.startswith(prefixes) and not any(c.isspace() for c in x):mapping[x]='id:'+digest(x)[:24]
    collect(graph)
    def remap(x):
        if isinstance(x,dict):return {mapping.get(k,k):remap(v) for k,v in x.items()}
        if isinstance(x,list):return [remap(v) for v in x]
        if isinstance(x,str):return mapping.get(x,x)
        return x
    g=remap(graph)
    # Hashes that attest the renamed representation must be regenerated.
    facts={f['fact_id']:f for f in g['facts']}
    for b in g.get('name_bindings',[]):b['source_fact_digest']=digest(facts[b['source_fact_id']])
    audio=g.get('audio_memory')
    if isinstance(audio,dict) and 'artifact_id' in audio:
        audio['artifact_id']=digest({k:v for k,v in audio.items() if k!='artifact_id'})
    # Candidate extraction history is an offline audit, not runtime memory.
    g.pop('name_grounding',None)
    g=TemporalEvidenceHypergraph(g).snapshot()
    return g,mapping
