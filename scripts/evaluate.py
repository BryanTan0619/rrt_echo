#!/usr/bin/env python3
"""Score predictions on BindScope paired units.

    python scripts/evaluate.py results/predictions                 # every <variant>.jsonl in the folder
    python scripts/evaluate.py runs/full/predictions.jsonl runs/binary/predictions.jsonl --reference full
    python scripts/evaluate.py results/predictions --subset complementary --latex

A predictions file has one {"question_id", "answer"} per line. The denominator is fixed to all paired
units in the dataset: a missing or unparseable answer counts as wrong. For every variant other than the
reference, the PairAcc / QAcc / BIR differences are bootstrapped over videos (10,000 resamples), with
Holm correction across the compared variants.

  QAcc    = mean correctness over the 2|U| probes
  PairAcc = fraction of units with both probes correct
  BIR     = sum_u |c_u1 - c_u2| / sum_u max(c_u1, c_u2)
"""
import argparse
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPES = [f'R{i}' for i in range(8)]


def load_dataset(path):
    qs = [json.loads(line) for line in open(path) if line.strip()]
    units = collections.defaultdict(list)
    for q in qs:
        units[q['unit_id']].append(q)
    units = {u: v for u, v in units.items() if len(v) == 2}
    return units


def load_predictions(path):
    return {r['question_id']: str(r.get('answer')) for r in map(json.loads, open(path)) if r.get('question_id')}


def counts(correct, units, uu):
    k = collections.Counter(sum(correct.get(q['question_id'], False) for q in units[u]) for u in uu)
    return k[2], k[1], len(uu)


def metrics(both, one, n):
    qacc = 100 * (2 * both + one) / (2 * n)
    pair = 100 * both / n
    bir = 100 * one / (both + one) if both + one else float('nan')
    return qacc, pair, bir


def bootstrap(ca, cb, units, uu, B=10000, seed=0):
    """Video-clustered bootstrap of metric(a) - metric(b); returns [(diff, lo, hi, p)] for QAcc, PairAcc, BIR."""
    by_video = collections.defaultdict(list)
    for u in uu:
        by_video[units[u][0]['video_id']].append(u)
    vids = sorted(by_video)
    per = {v: (counts(ca, units, by_video[v]), counts(cb, units, by_video[v])) for v in vids}

    def diff(sel):
        x, y = [0, 0, 0], [0, 0, 0]
        for v in sel:
            for i in range(3):
                x[i] += per[v][0][i]
                y[i] += per[v][1][i]
        return [p - q for p, q in zip(metrics(*x), metrics(*y))]

    obs = diff(vids)
    rnd = random.Random(seed)
    draws = [diff([vids[rnd.randrange(len(vids))] for _ in vids]) for _ in range(B)]
    out = []
    for i in range(3):
        d = sorted(r[i] for r in draws)
        p = min(1.0, 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / B)
        out.append((obs[i], d[int(0.025 * B)], d[int(0.975 * B) - 1], p))
    return out


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    adj, run = {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, pvals[k] * (len(order) - j)))
        adj[k] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('predictions', nargs='+', help='prediction files, or one folder of <variant>.jsonl')
    ap.add_argument('--dataset', default=str(ROOT / 'data/bindscope.jsonl'))
    ap.add_argument('--reference', default='full', help='variant the others are compared with')
    ap.add_argument('--subset', choices=['all', 'complementary', 'tf_minimal', 'paraphrase', 'true_false', 'multiple_choice'], default='all')
    ap.add_argument('--videos', nargs='*', help='restrict to these videos (e.g. the example video)')
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--latex', action='store_true', help='also print rows in the paper table layout')
    a = ap.parse_args()

    units = load_dataset(a.dataset)
    paths = sorted(Path(a.predictions[0]).glob('*.jsonl')) if Path(a.predictions[0]).is_dir() else [Path(p) for p in a.predictions]
    name = lambda p: p.parent.name if p.name == 'predictions.jsonl' else p.stem
    correct = {}
    for p in paths:
        pred = load_predictions(p)
        correct[name(p)] = {q['question_id']: pred.get(q['question_id']) == str(q['answer']) for v in units.values() for q in v}
    names = sorted(correct, key=lambda n: (n != a.reference, n))

    uu = sorted(u for u in units if not a.videos or units[u][0]['video_id'] in a.videos)
    if a.subset in ('complementary', 'tf_minimal', 'paraphrase'):
        kind = json.load(open(ROOT / 'results/pair_kind.json'))
        uu = [u for u in uu if kind.get(u) == a.subset]
    elif a.subset != 'all':
        uu = [u for u in uu if units[u][0]['question_type'] == a.subset]
    print(f'{len(uu)} paired units, {len({units[u][0]["video_id"] for u in uu})} videos, subset={a.subset}\n')

    print(f'{"variant":26s} {"QAcc":>6s} {"PairAcc":>8s} {"BIR":>6s}')
    for n in names:
        print(f'{n:26s} {"%6.1f %8.1f %6.1f" % metrics(*counts(correct[n], units, uu))}')

    if a.reference in correct and len(names) > 1:
        others = [n for n in names if n != a.reference]
        res = {n: bootstrap(correct[a.reference], correct[n], units, uu, a.bootstrap) for n in others}
        for i, m in enumerate(('QAcc', 'PairAcc', 'BIR')):
            adj = holm({n: res[n][i][3] for n in others})
            print(f'\n{a.reference} minus variant, {m}: diff [95% CI] p (Holm)')
            for n in others:
                d, lo, hi, p = res[n][i]
                print(f'  {n:24s} {d:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]  p={p:.4f} ({adj[n]:.4f})')

    print('\nPairAcc by binding type: ' + ' '.join(f'{t:>6s}' for t in TYPES))
    for n in names:
        row = []
        for t in TYPES:
            ut = [u for u in uu if units[u][0]['binding_type'] == t]
            row.append('%6.1f' % metrics(*counts(correct[n], units, ut))[1] if ut else '    --')
        print(f'  {n:24s} ' + ' '.join(row))

    if a.latex:
        print()
        for n in names:
            print(f'        {n}')
            for i, lab in enumerate(('QAcc $\\uparrow$   ', 'PairAcc $\\uparrow$', 'BIR $\\downarrow$  ')):
                cells = []
                for t in TYPES:
                    ut = [u for u in uu if units[u][0]['binding_type'] == t]
                    cells.append('%.1f' % metrics(*counts(correct[n], units, ut))[i] if ut else '--')
                print(f'            & {lab} & {metrics(*counts(correct[n], units, uu))[i]:.1f}\n            & ' + ' & '.join(cells) + ' \\\\')


if __name__ == '__main__':
    main()
