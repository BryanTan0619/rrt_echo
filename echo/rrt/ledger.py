"""Build representations from a frozen RRT ledger, never from a Full artifact.

Normalization/identity admission is shared upstream. Temporal event organization
is constructed independently downstream. Text uses an explicit primitive-field
allowlist, never a dump of materialized graph metadata.
"""
import copy, json
from .memory import RRTMemory
from .audio import attach_audio
from .name_grounding import build_name_registry, annotate_graph
from .hypergraph import TemporalEvidenceHypergraph, ALIGNMENT_KEY, alignment_policy
from .representation_views import neutralize_ids
from ..schema import digest

FACT_FIELDS=('fact_id','kind','predicate','roles','value','extent','observed_times','observed_segments','joint_evidence','evidence_by_slot','unresolved_slots','carrier','binding_admission')


def memory(bundle):
    assert bundle['schema']=='echo.common_rrt.v1'
    m=RRTMemory(bundle['video_id']);m.observations=copy.deepcopy(bundle['observations'])
    m.journal=copy.deepcopy(bundle['identity_journal']);m.text_revisions=copy.deepcopy(bundle['text_revisions'])
    m.binding_admission=copy.deepcopy(bundle.get('binding_admission'))
    return m

def grounded_names(graph,bundle):
    if bundle.get('binding_admission'):
        from .binding_admission import admitted_name_registry
        return admitted_name_registry(graph,bundle['binding_admission'])
    return build_name_registry(graph)


def primitive_source(bundle):
    # RRT normalization commits the same observations and identity decisions for
    # every variant. Only these primitive fields may cross into flat text.
    g=memory(bundle).snapshot()
    reg,names=grounded_names(g,bundle)
    g=annotate_graph(g,names,reg)
    return g


def build_graph(bundle,arm):
    if arm!='hypergraph':raise ValueError('graph arm required')
    g=memory(bundle).materialize() # Full organization from original ledger
    g=attach_audio(g,bundle['audio_memory'])
    reg,names=grounded_names(g,bundle);g=annotate_graph(g,names,reg)
    g[ALIGNMENT_KEY]=alignment_policy(1.5)
    g=TemporalEvidenceHypergraph(g).snapshot()
    # Canonical JSON types in every variant: tuple-valued identity evidence must
    # not bypass the recursive ID renaming used by the runtime.
    g=json.loads(json.dumps(g))
    g,_=neutralize_ids(g)
    return g
