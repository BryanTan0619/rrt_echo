"""Query-blind, evidence-carrying local names; a shared string is not identity.

Caption/OCR extraction creates mentions. Admission requires an explicit local
name-carrier statement or a recorded visual review. No new same-identity edges
are ever inferred from names. Existing identity decisions are left untouched.
"""
from __future__ import annotations
import collections
import copy
import re
from ..schema import digest

METHOD = 'local_name_evidence_v2'


def normalize_name(value):
    return re.sub(r'\s+', ' ', str(value).strip()).title()


def is_person_name(value):
    """Candidate shape only; caller MUST still validate the local name relation."""
    value = str(value).strip()
    tokens = re.findall(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", value, re.UNICODE)
    return (0 < len(value) <= 48 and 0 < len(tokens) <= 4
            and ' '.join(tokens).casefold() == value.casefold())


def evidence_frames(fact):
    return set(fact.get('joint_evidence') or []) | {
        m for ms in (fact.get('evidence_by_slot') or {}).values() for m in ms}


def caption_names_person(description, value):
    """An explicit PERSON-name relation in the *unannotated* local caption.

    Mere occurrence of a token on clothes or elsewhere in a caption is not
    enough: brands and locations also appear on clothing. Ambiguous cases are
    retained for visual review. This is corroboration within frozen RRT, not an
    independent perception measurement.
    """
    n = re.escape(str(value).strip())
    q = r'[\s\"\'‘’“”]*'
    end = r'(?![\w\-])'
    patterns = [r'\b(?:name|name\s*tag|name\s*badge)\s*(?:is|reads?|says?|of)?'+q+n+end,
                r'\b(?:named|identified as)'+q+n+end]
    return any(re.search(p, description, re.I) for p in patterns)


def name_candidates(graph):
    instances = {i['instance_id']: i for i in graph.get('instances', [])}
    rows = []
    for f in graph.get('facts', []):
        if f.get('kind') != 'text' or not f.get('value'):
            continue
        iid = (f.get('roles') or {}).get('owner')
        inst = instances.get(iid, {})
        if inst.get('kind') != 'person':
            continue
        value = str(f['value']).strip()
        shared = evidence_frames(f) & {r['media_id'] for r in inst.get('regions', [])}
        shared &= graph.get('evidence', {}).keys()
        row = {'mention_id': 'name-mention:'+digest([f['fact_id'],iid,value])[:24],
               'text':value,'name':normalize_name(value),'entity_type':'unresolved',
               'source_fact_id':f['fact_id'],'source_fact_digest':digest(f),
               'person_instance':iid,'owner_description':inst.get('description',''),
               'carrier':f.get('carrier','unknown'),'evidence_ids':sorted(shared),
               'observed_times':list(f.get('observed_times') or []),'status':'unresolved'}
        if not is_person_name(value):
            row.update(status='rejected',reason='non_person_text_or_not_name_candidate')
        elif not shared:
            row['reason']='no_shared_owner_text_evidence'
        elif f.get('carrier') == 'overlay':
            row['reason']='overlay_owner_requires_visual_review'
        elif caption_names_person(inst.get('description',''),value):
            row.update(status='accepted',entity_type='PERSON',reason='explicit_local_caption_name_relation',
                       verification='frozen_rrt_caption_corroboration_not_independent',media_verified=False)
        else:
            row['reason']='mention_does_not_establish_person_name_or_owner'
        rows.append(row)
    return rows


def build_name_registry(graph, min_confidence_frames=1, *, reviews=()):
    """Compatible return API; registry contains only locally grounded bindings.

    Optional visual reviews are tied to the exact source fact digest and owner,
    cite actually presented source media, and make a narrow local claim only.
    """
    if graph.get('name_grounding',{}).get('method') not in (None,METHOD):
        raise ValueError('legacy_name_grounding_requires_clean_source_snapshot')
    reviewed = {r.get('mention_id'): r for r in reviews}
    rows = name_candidates(graph)
    for row in rows:
        r = reviewed.get(row['mention_id'])
        if not r or row['status']=='rejected':
            continue
        checks = ('name_is_person','names_this_person','owner_distinguished','not_brand_or_program')
        cited=set(r.get('evidence_ids') or []); shown=set(r.get('shown_media_ids') or [])
        if (r.get('source_fact_digest')==row['source_fact_digest']
                and r.get('person_instance')==row['person_instance'] and r.get('query_blind') is True
                and r.get('verdict')=='supported' and all(r.get(k) is True for k in checks)
                and cited and cited <= set(row['evidence_ids']) and cited <= shown
                and r.get('request_sha256') and r.get('model')):
            row.update(status='accepted',entity_type='PERSON',reason='local_visual_name_review',
                       verification='model_visual_review_not_human_verified',media_verified=False,
                       review=copy.deepcopy(r))
    groups=collections.defaultdict(list)
    for r in rows:
        if r['status']=='accepted' and len(r['evidence_ids'])>=min_confidence_frames:
            groups[r['person_instance']].append(r)
    # Conflicting local names remain unresolved; frequency cannot break the tie.
    selected=[]
    for iid,rs in groups.items():
        if len({r['name'].casefold() for r in rs})==1:selected.extend(rs)
    registry={}
    for r in selected:
        item=registry.setdefault(r['name'],{'instances':[],'entities':[], 'occurrences':0,
                                          'frames':[], 'evidence_fact_ids':[], 'bindings':[]})
        b={k:copy.deepcopy(r[k]) for k in ['name','person_instance','source_fact_id','source_fact_digest',
                                         'evidence_ids','observed_times','verification','media_verified']}
        b.update(binding_id='name-binding:'+digest(b)[:24],status='accepted',relation='names_local_person',
                 entity_type='PERSON',scope='local_instance_only_not_global_identity')
        if 'review' in r:b['review']=copy.deepcopy(r['review'])
        item['bindings'].append(b);item['occurrences']+=1
        for key,values in [('instances',[r['person_instance']]),('frames',r['evidence_ids']),('evidence_fact_ids',[r['source_fact_id']])]:
            item[key]=sorted(set(item[key])|set(values))
    return registry,{r['person_instance']:r['name'] for r in selected}


def annotate_graph(graph, instance2name, registry=None, merge_identities=False):
    """Store typed local name evidence separately; never rewrite raw captions."""
    if graph.get('name_grounding',{}).get('method') not in (None,METHOD):
        raise ValueError('legacy_name_grounding_requires_clean_source_snapshot')
    g=copy.deepcopy(graph);registry=registry or {}
    bindings=[copy.deepcopy(b) for r in registry.values() for b in r.get('bindings',[])
              if instance2name.get(b['person_instance'])==b['name']]
    g['name_bindings']=sorted(bindings,key=lambda b:b['binding_id'])
    g['name_grounding']={'method':METHOD,'identity_merges':0,'description_rewrites':0,
                         'policy':'explicit local name evidence; same string never merges identities',
                         'accepted_bindings':len(bindings),'accepted_instances':len(instance2name),
                         'candidate_mentions':name_candidates(graph),
                         'legacy_merge_argument_ignored':bool(merge_identities)}
    g.pop('snapshot_id',None);g['snapshot_id']=digest(g)
    return g


