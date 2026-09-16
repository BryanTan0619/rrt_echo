"""Version-controlled prompts with separated observation and identity authority."""

LOCAL = """Inspect only these chronological video frames. Extract local visual facts,
not a story summary. You are not given global names, family roles, historical
captions or questions. Use neutral packet-local IDs for all people and objects.
Instance kind must be exactly person, object, or region. Adults and infants are
person; a visible body-only individual is also person. A detached/ambiguous arm
without a grounded individual is region, not an invented full person. Reuse one
local person ID across frames only when visible continuity supports it. Describe
each person independently of detector track fragments. A detector may miss people.
Detector boxes and track labels are region proposals, not identity truth. You may
add a missed person or split a composite adult/child box, but only cite visible
regions; if grounding is unclear preserve the missing slot instead of fabricating
a box. A body-only person is valid. Different local IDs are not proof of different
global people. For each separate occurrence jointly bind predicate and directed
roles. Preserve small objects, contacts, transfers, injuries and visible text.
Text-on-a-body is not evidence of writing or of its author. Bind text's owner and
region separately. Do not infer a state transition from a single after-state.
Do not infer persistence across unobserved frames. All evidence IDs must be among
the supplied media IDs. Never extend or extrapolate frame ID sequences.
Use compact JSON. Cite 1-4 most informative regions per instance, enough to cover
its event roles, rather than listing every frame. Avoid duplicate facts. Boxes are normalized xyxy in [0,1]. For every fact provide
predicate evidence, per-role evidence, and joint_evidence of this occurrence.
Every roles value MUST reference an instance_id defined in instances. Do not put
text, descriptions, or an undeclared screen/hand into roles. Visible screen text
uses a grounded region instance as owner and the exact string goes in value.
If no observation is grounded, return empty instances and facts. Do not mark a
patient missing for an intrinsically unary action. All role/predicate slot evidence
must include the frames containing the participating instance regions.
Return only JSON with keys instances and facts. Example schema (not scene facts):
{"instances":[{"instance_id":"p1","kind":"person","description":"visible appearance",
"track_ref":null,"regions":[{"media_id":"supplied-id","box":[0.1,0.1,0.4,0.9]}]}],
"facts":[{"fact_id":"e1","kind":"event","predicate":"reach",
"roles":{"agent":"p1"},"evidence_by_slot":{"predicate":["supplied-id"],
"agent":["supplied-id"]},"joint_evidence":["supplied-id"],
"observed_media_ids":["supplied-id"],"unresolved_slots":["patient"]}]}
For kind state/attribute/text, predicate is the property, roles must include owner,
value contains the observed value, and evidence_by_slot must also include value.
Do not output identity decisions, global entities, confidence-based commitments or
time_bounds. Actual observed times are recovered from media, not guessed by you.
"""

IDENTITY = """Compare the explicitly marked local instance endpoints using the actual
reference and current frames/regions. Neutral IDs are not names. Report same,
different or unresolved. Direct visual evidence can independently support same;
no other modality must agree. Clothing, shared objects, narrative or assumed
family roles alone do not establish identity. Account for competing candidates,
occlusion and appearance changes. Duplicate detections, mirrors and screens are
not automatically different people. Cite media for BOTH endpoints and describe
observable correspondences and counterevidence. Do not change event roles.
Return JSON: {"verdict":"same|different|unresolved","media_ids":["..."],
"visual_basis":"visible evidence and relevant ambiguity"}.
"""

ANSWER = """Answer the multiple-choice question using only the supplied graph payload.
You must choose one listed option even if evidence is insufficient. Do not use
external video knowledge. Local-joint support does not imply global identity,
owner, text or duration support. Missing evidence does not prove a negative.
Only explicit evidence-scoped joins are allowed. Return JSON {"answer":"option"}.
"""

AUDIT = """Audit the OUTPUT ANSWER, not the question proposition, using only the exact
reader graph payload. Split substantive claims into assertions; all must have
support for the answer to be supported. Check identity, roles, owner, characters
and temporal scope. A negative answer without counterevidence/completeness proof
is insufficient. Do not use unseen evidence or story knowledge. Return JSON
{"assertions":[{"assertion":"...","support_verdict":"supported|contradicted|insufficient",
"evidence_chain":["received fact/identity/relation ID"]}]}.
This is graph support, not an independent verification of video truth.
"""


JOINT_LOCAL = """Observe this ordered visual segment once and return a compact bound
observation: instances and facts together. Use minified JSON, no indentation.
Local IDs must be i0 through i11. Allocate each once and reuse it in roles.
Literal attributes, colors and OCR strings go in value, NEVER in roles.
For example an attribute uses roles={"owner":"i0"}, value="observed attribute".
Do not make a separate fact for each frame of the same observation; combine its
source frame IDs in one fact. Describe what is visible neutrally;
do not start from a proposed story or assume a detected region is a person.
All supplied frames belong to this observation. Each fact must cite its actual
supporting frames, not every frame merely because it was supplied.
Create one local ID per visibly continuous person/object, not per frame or box.
Use person, object, or region. A hand/arm without a visible owner is a region,
not a new person. Keep adult and infant separate, including inside a composite
box. Detector labels are proposals only. Describe each instance once, briefly.
Cite up to four grounded regions, INTEGER xyxy coordinates in 0..1000.
While describing an action, bind its predicate and directed roles to these IDs
in the same response. Separate writer, written-on region, and its owner. Record
an observed part_of relation only when its region/person connection is visible;
an unknown owner does not erase a visible action. Never infer ownership from
adjacency or a shared object alone. Do not create global names or identity merges.
Keep distinct occurrences separate. Retain small-object actions, contacts,
changes, visible attributes and legible text. Do not fill output with generic
textures, repeated character descriptions, or paraphrases of the same fact.
A repeated observation can add frame evidence without another duplicate fact.
For events use action roles; for state/attribute/text include owner and value.
Every role value references an instance defined in this response. For each fact
provide evidence_by_slot for predicate, every role, and value when applicable;
joint_evidence is the frame evidence of the joint action/roles. Cite participant
regions on the relevant supporting frames. Keep missing bindings in unresolved_slots.
Do not infer continuous states or a transition from a single after-state. Text
seen on skin does not prove who wrote it. Do not hallucinate exact OCR characters.
No prose caption, explanations, confidence scores, or full story summary outside
this structure. Return up to 12 instances and 20 nonduplicate facts. These are
transport limits, not a request to fill every slot. Use only supplied frame IDs.
"""
