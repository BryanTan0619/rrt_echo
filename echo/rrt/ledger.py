"""Build the ECHO memory of one video from its cached observations.

The observation ledger (VLM window observations, cross-window identity journal, ASR) is normalised into
facts and identities, organised into the temporal evidence hypergraph, and serialised for retrieval.
"""
import copy, json
from .memory import RRTMemory
from .audio import attach_audio
from .name_grounding import build_name_registry, annotate_graph
from .hypergraph import TemporalEvidenceHypergraph, ALIGNMENT_KEY, alignment_policy
from .representation_views import neutralize_ids
from ..schema import digest

def memory(bundle):
    assert bundle['schema']=='echo.common_rrt.v1'
    m=RRTMemory(bundle['video_id']);m.observations=copy.deepcopy(bundle['observations'])
    m.journal=copy.deepcopy(bundle['identity_journal']);m.text_revisions=copy.deepcopy(bundle['text_revisions'])
    m.binding_admission=copy.deepcopy(bundle.get('binding_admission'))
    return m

def grounded_names(graph,bundle):
    return build_name_registry(graph)


def build_memory(bundle):
    """Observation bundle (observations.json) -> ECHO memory (hypergraph.json)."""
    g=memory(bundle).materialize()
    g=attach_audio(g,bundle['audio_memory'])
    reg,names=grounded_names(g,bundle);g=annotate_graph(g,names,reg)
    g[ALIGNMENT_KEY]=alignment_policy(1.5)
    g=TemporalEvidenceHypergraph(g).snapshot()
    # Canonical JSON types: tuple-valued identity evidence must not bypass the
    # recursive ID renaming.
    g=json.loads(json.dumps(g))
    g,_=neutralize_ids(g)
    return g
