"""Tool definitions (OpenAI function-calling format): the single retrieval tool `search` and `submit_answer`."""

SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": 'Search the video memory. `query` ranks by whole-sentence meaning (embedding); every other parameter is an exact keyword filter. Keyword syntax: space = AND, | = OR, "quoted phrase"; English matches word starts (jump → jumps, jumping), Chinese matches substrings. Line: id [time] window kind "predicate" = "value" | role=ENTITY "description" ; … | ⚠ flags. Results end with anchors to reuse.',
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Sentence to rank by meaning (any language). Optional.",
                },
                "kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["event", "text", "state", "attribute", "speech"],
                    },
                },
                "predicate": {
                    "type": "string",
                    "description": "Keywords in the action/property, e.g. 'jump|leap'.",
                },
                "value": {
                    "type": "string",
                    "description": "Keywords in the value: on-screen text, names, colours, states.",
                },
                "roles": {
                    "type": "object",
                    "description": "Slot → filler. Slot: role name, @actor (doer), @target (acted on / recipient), @context (tool, surface), @owner, or *. Filler: entity id or list of ids, keywords matched in that slot's description (e.g. 'woman blond|blonde'), or '?' (slot filled).",
                },
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these fact ids; with view=entity, entity ids.",
                },
                "window": {"description": "Window number(s), e.g. 'W17' or ['W4','W5']."},
                "start": {"type": "string", "description": "mm:ss or seconds."},
                "end": {"type": "string"},
                "around": {
                    "type": "string",
                    "description": "'mm:ss' or 'mm:ss±sec' (default ±15).",
                },
                "period": {
                    "type": "string",
                    "description": "beginning, end, first_half, second_half, first/middle/last_third, first/second/third/last_quarter, or 'k/n'.",
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "time_asc", "time_desc"],
                    "description": "Default relevance with query, else time_asc.",
                },
                "limit": {"type": "integer", "description": "Default 10, max 100."},
                "offset": {"type": "integer"},
                "view": {
                    "type": "string",
                    "enum": ["list", "window", "closure", "entity"],
                    "description": "list (default); window = every record of the matched windows (or of `window` alone); closure = records the original system bundles with each hit; entity = profiles of entity ids in `ids` (two ids → same-person check incl. whether both appear in one window).",
                },
                "include_dropped": {
                    "type": "boolean",
                    "description": "Also return records the original validator dropped (weak evidence).",
                },
            },
        },
    },
}

SUBMIT_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final answer. Call exactly once, at the end.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The option letter (A/B/C/D) or True/False.",
                },
                "confidence": {
                    "type": "number",
                    "description": "0–1: how strongly the memory supports this option over the others.",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fact ids the answer relies on.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Short justification, including which assumptions (e.g. identity of a named person) you made.",
                },
            },
            "required": ["answer", "confidence", "reasoning"],
        },
    },
}

TOOL_SPECS = [SEARCH_SPEC, SUBMIT_SPEC]
