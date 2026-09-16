"""Shortest accepted witness paths into a frozen entity version."""

from collections import defaultdict, deque


def identity_paths(graph, endpoints):
    adjacency = defaultdict(list)
    for row in graph["identity_decisions"]:
        if row["status"] != "accepted" or row["proposal"]["verdict"] != "same":
            continue
        p = row["proposal"]
        adjacency[p["left"]].append((p["right"], row["decision_id"]))
        adjacency[p["right"]].append((p["left"], row["decision_id"]))
    entity_of = {i: e for e in graph["entities"] for i in e["local_instances"]}
    paths = {}
    for source in endpoints:
        entity = entity_of[source]
        root = min(entity["local_instances"])
        queue = deque([(source, [])])
        seen = {source}
        found = None
        while queue:
            node, path = queue.popleft()
            if node == root:
                found = path
                break
            for nxt, did in sorted(adjacency[node]):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, path + [did]))
        if found == [] and adjacency[source]:
            # A representative can show one membership witness rather than
            # all unrelated members of its component.
            found = [sorted(adjacency[source])[0][1]]
        paths[source] = found or []
    return paths
