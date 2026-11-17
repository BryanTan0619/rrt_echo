#!/usr/bin/env python3
"""Build the memory variants from cached observations.

    python scripts/build_memory.py --observations data/example --out memory [--variants full binary ...]

Reads <observations>/<video_id>/observations.json and writes <out>/<variant>/<video_id>/hypergraph.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo.variants import VARIANTS, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--observations', default=str(ROOT / 'data/example'))
    ap.add_argument('--out', default=str(ROOT / 'memory'))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument('--videos', nargs='*', help='default: every folder under --observations')
    a = ap.parse_args()
    src = Path(a.observations)
    videos = a.videos or sorted(p.name for p in src.iterdir() if (p / 'observations.json').exists())
    for v in videos:
        bundle = json.loads((src / v / 'observations.json').read_text())
        for name, memory in build(bundle, a.variants).items():
            d = Path(a.out) / name / v
            d.mkdir(parents=True, exist_ok=True)
            (d / 'hypergraph.json').write_text(json.dumps(memory))
        print(v, 'built', ', '.join(a.variants), flush=True)


if __name__ == '__main__':
    main()
