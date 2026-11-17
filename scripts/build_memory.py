#!/usr/bin/env python3
"""Build the ECHO memory from cached observations.

    python scripts/build_memory.py --observations data/example --out memory

Reads <observations>/<video_id>/observations.json and writes <out>/<video_id>/hypergraph.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo import build_memory  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observations', default=str(ROOT / 'data/example'))
    ap.add_argument('--out', default=str(ROOT / 'memory'))
    ap.add_argument('--videos', nargs='*', help='default: every folder under --observations')
    a = ap.parse_args()
    src = Path(a.observations)
    videos = a.videos or sorted(p.name for p in src.iterdir() if (p / 'observations.json').exists())
    for v in videos:
        memory = build_memory(json.loads((src / v / 'observations.json').read_text()))
        d = Path(a.out) / v
        d.mkdir(parents=True, exist_ok=True)
        (d / 'hypergraph.json').write_text(json.dumps(memory))
        print(v, 'facts', len(memory['facts']), 'hyperedges', len(memory['event_hyperedges']), flush=True)


if __name__ == '__main__':
    main()
