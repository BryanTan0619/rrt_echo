"""Task instructions only. Field constraints live in wire.py."""

VERSION = "echo-perception-v1.1-wire-guidance"
JOINT_LOCAL = """Read the sequence as a continuous visual clip.
Extract entities and distinct events in TARGET; jointly bind actions to directed roles.
Reuse local IDs when visible continuity supports them. Distinguish entities from parts.
Region and track proposals are hints, not identity decisions.
Use CONTEXT and REFERENCE only to propose explicit correspondence or ownership links;
do not record their appearances as TARGET observations.
Cite the supplied evidence for each binding. Leave missing bindings unresolved.
Record observed states without filling temporal gaps; distinguish visible text from authorship.
Return minified JSON only: no markdown, caption, or repeated descriptions.
Use at most four representative region anchors per instance; never enumerate every frame or repeat an evidence ID.
State, attribute, and text facts require roles.owner and a non-null value.
For each fact, evidence_by_slot contains predicate, every role, and value for non-events.
Role evidence must anchor its instance region; all slot and observed evidence must be included in joint_evidence.
"""
# Alias for integration: use the same single-call task, not a second extraction pass.
LOCAL = JOINT_LOCAL
