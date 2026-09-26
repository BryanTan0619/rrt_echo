"""Export the exact BindScope sources used by the full ECHO evaluation.
Only allowlisted public QA fields are copied; source/private annotations are not.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA = Path('/apdcephfs/11/apdcephfs_nj7/share_303382070/xdata_public/Binding_Dataset')
ROOT = Path(__file__).resolve().parents[1]


def export(out, data_root=DATA):
    files = {
        'binding_pairs': data_root/'bindscope_pairs_mrx_v1.5_release_0924/bindscope_pairs_mrx_v1.5_reviewed.json',
        'singles': data_root/'bindscope_pairs_mrx_v1.5_release_0924/bindscope_singles_v1.5.json',
        'presence_pairs': data_root/'bindscope_b0.json',
    }
    rows = []
    reviews = Counter()
    for kind in ('binding_pairs', 'singles'):
        source = json.loads(files[kind].read_text())
        for sample in source['samples']:
            paired = kind == 'binding_pairs'
            questions = sample['questions']
            assert len(questions) == (2 if paired else 1), sample['pair_id']
            review = sample.get('quality', {}).get('review_status_0924', {})
            reviews[kind+':human_reviewed'] += bool(review.get('human_reviewed'))
            reviews[kind+':not_marked_human_reviewed'] += not bool(review.get('human_reviewed'))
            for index, question in enumerate(questions, 1):
                raw_choices = question['choices']
                key = sample['private_annotation']['gold_answers'][question['question_id']]
                assert key in raw_choices
                if len(raw_choices) == 2:
                    assert {v.capitalize() for v in raw_choices.values()} == {'True','False'}
                    choices = {'True':'True','False':'False'}
                    answer = raw_choices[key].capitalize()
                    question_type = 'true_false'
                else:
                    assert set(raw_choices) == {'v0','v1','v2','v3'}
                    choices = {chr(65+i): raw_choices[f'v{i}'] for i in range(4)}
                    answer = chr(65+int(key[1:]))
                    question_type = 'multiple_choice'
                rows.append({
                    'question_id': question['question_id'], 'video_id': sample['video_id'],
                    'binding_type': sample['binding_type'], 'unit_id': sample['pair_id'],
                    'unit_type': 'binding_pair' if paired else 'single',
                    'pair_id': sample['pair_id'] if paired else None,
                    'pair_index': index if paired else None, 'unit_size': len(questions),
                    'direction': question.get('direction', 'single' if not paired else None),
                    'question_type': question_type, 'question': question['text'],
                    'options': choices, 'answer': answer,
                })
    for line in files['presence_pairs'].read_text().splitlines():
        if not line.strip(): continue
        statement = json.loads(line)
        answer = str(statement['answer']).capitalize()
        assert answer in ('True','False')
        rows.append({
            'question_id': statement['statement_id'], 'video_id': statement['video_id'],
            'binding_type': 'R0', 'unit_id': statement['pair_id'],
            'unit_type': 'presence_pair', 'pair_id': statement['pair_id'],
            'pair_index': None, 'unit_size': 2, 'direction': None,
            'question_type': 'true_false', 'question': statement['text'],
            'options': {'True':'True', 'False':'False'}, 'answer': answer,
        })
    grouped = defaultdict(list)
    for row in rows: grouped[row['unit_id']].append(row)
    for unit, group in grouped.items():
        assert len(group) == group[0]['unit_size'], unit
        assert len({(x['video_id'],x['binding_type'],x['unit_type']) for x in group}) == 1, unit
        if group[0]['unit_type'] == 'presence_pair':
            assert sorted(x['answer'] for x in group) == ['False','True'], unit
            for index,row in enumerate(group,1):row['pair_index'] = index
    assert len({x['question_id'] for x in rows}) == len(rows)
    assert len(rows)==2806 and len(grouped)==1557
    assert len({x['video_id'] for x in rows})==77
    for row in rows:
        assert row['answer'] in row['options']
        assert row['question'].strip()
        assert all(isinstance(x,str) and x.strip() for x in row['options'].values())
    questions_by_type = Counter(x['binding_type'] for x in rows)
    units_by_type = Counter(group[0]['binding_type'] for group in grouped.values())
    paired_by_type = Counter(group[0]['binding_type'] for group in grouped.values() if len(group)==2)
    singles_by_type = Counter(group[0]['binding_type'] for group in grouped.values() if len(group)==1)
    question_formats = Counter(x['question_type'] for x in rows)
    pair_formats = Counter(group[0]['question_type'] for group in grouped.values() if len(group)==2)
    assert all(len({x['question_type'] for x in g}) == 1 for g in grouped.values())
    out.mkdir(parents=True,exist_ok=True)
    public = out/'bindscope_qa.json'
    public.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    jsonl = out/'bindscope_qa.jsonl'
    jsonl.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    manifest = {
        'dataset': 'BindScope', 'release_source_version': 'mrx_v1.5_0924_with_current_R0',
        'status': 'prepared_release_candidate; source human audit incomplete',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'questions': len(rows), 'videos': len({x['video_id'] for x in rows}),
        'scoring_units': len(grouped), 'binding_pairs': 1110, 'presence_pairs': 139,
        'paired_units': 1249, 'single_units': 308,
        'questions_by_type':dict(sorted(questions_by_type.items())),
        'questions_by_format':dict(question_formats), 'paired_units_by_format':dict(pair_formats),
        'scoring_units_by_type':dict(sorted(units_by_type.items())),
        'paired_units_by_type':dict(sorted(paired_by_type.items())),
        'single_units_by_type':dict(sorted(singles_by_type.items())),
        'source_review_counts':dict(reviews),
        'count_difference_from_2810': 'Current R0 source contains 139 pairs / 278 questions, rather than the previous 141 pairs / 282 questions. No records were invented or restored from unaudited pools.',
        'sources':{k:{'path':str(v.relative_to(data_root)),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in files.items()},
        'outputs':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (public,jsonl)},
        'evaluation':{'question_accuracy':'mean individual question correctness',
                      'pair_accuracy':'both questions in each complete paired unit must be correct; exclude singles',
                      'paired_denominator_R1_R7':1110,'paired_denominator_R0_R7':1249,
                      'all_unit_denominator':1557,'question_denominator':2806,
                      'independent_random_baseline_four_option_pair':0.0625,
                      'independent_random_baseline_presence_pair':0.25,
                      'model_input':'one question and shuffled options per fresh conversation; do not expose answers, pair mates, directions or scoring metadata',
                      'answer_remapping':'After option shuffling, remap predictions to original semantic option IDs before scoring'},
        'validation':{'unique_question_ids':True,'correct_pair_sizes':True,'answers_in_options':True,
                      'pair_video_and_type_consistency':True,'R0_true_false_balance':True,
                      'private_annotations_excluded':True},
    }
    (out/'release_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines = ['# BindScope QA Release Files', '',
             'This directory contains the v1.5 QA export used by the current ECHO full-dataset evaluation. Distribution of these files does not imply that all source annotations have completed human review.', '',
             '- `bindscope_qa.json`: a standard JSON array, with one record per question.',
             '- `bindscope_qa.jsonl`: the same records in JSON Lines format.',
             '- `release_manifest.json`: dataset statistics, source hashes, validation results, and the evaluation protocol.', '',
             'The export contains **77 videos, 2,806 questions, and 1,557 scoring units**: 1,110 R1–R7 pairs (2,220 questions), 139 R0 pairs (278 questions), and 308 single questions.',
             f'By format, there are {question_formats["multiple_choice"]} multiple-choice questions and {question_formats["true_false"]} true/false questions. Paired units include {pair_formats["multiple_choice"]} multiple-choice pairs and {pair_formats["true_false"]} true/false pairs. R1–R7 also includes true/false questions.', '',
             'The earlier count of 2,810 questions assumed 141 R0 pairs. The current R0 source contains 139 pairs, accounting for the four-question difference. No questions were added to reach the earlier count.', '',
             '| Type | Questions | Paired units | Single units | Total units |',
             '|---|---:|---:|---:|---:|']
    for typ in sorted(questions_by_type):
        lines.append(f'| {typ} | {questions_by_type[typ]} | {paired_by_type[typ]} | {singles_by_type[typ]} | {units_by_type[typ]} |')
    lines += ['', '## Fields and Evaluation', '',
              '`question_id` is a stable question identifier, `video_id` identifies the video, and `binding_type` takes values R0–R7. For multiple-choice questions, `options` uses keys A–D and `answer` is the correct letter. For true/false questions, option keys and answers are the strings `True` and `False`.', '',
              '`unit_id` identifies the scoring unit; `unit_type` is `binding_pair`, `presence_pair`, or `single`. Paired questions share a `pair_id` and have `pair_index` 1 or 2; both fields are null for singles. A pair is correct only when both questions are answered correctly. Singles are reported separately and excluded from PairAcc.', '',
              'R1–R7 PairAcc has a denominator of 1,110; PairAcc including R0 has a denominator of 1,249; All-Unit Accuracy has a denominator of 1,557. Independent uniform guessing gives 6.25% for a pair of four-option questions and 25% for a pair of true/false questions. Model errors across two questions may be correlated, so these baselines do not determine observed text-only scores.', '',
              'Evaluate each question in a fresh conversation, randomize option positions, and map predictions back to the original options before scoring. Model inputs should contain only the video (or memory allowed by the method), the current question, and its options. Do not provide gold answers, the other question in the pair, `direction`, or scoring metadata.', '',
              '`direction` preserves the source label; it does not guarantee that every pair contains semantically opposite role queries. The QA files exclude private latent bindings, evidence intervals, source questions, text-only test scores, and reviewer information.', '',
              '## Remaining Human Review', '',
              'Although the source filenames contain "reviewed", some records are not marked as human-reviewed. The counts below reflect source metadata and are not a guarantee of final annotation quality:', '']
    for k,v in sorted(reviews.items()):lines.append(f'- `{k}`: {v}')
    lines += ['', 'Human review should be completed before these records are presented as a fully validated benchmark. This export checks fields and structural consistency; it does not independently rewatch videos to verify answer correctness.', '',
              'Reproduce the export with `python tools/export_bindscope_release.py --data-root /path/to/Binding_Dataset --out bindscope`. The exporter preserves the original source files and does not affect running experiments.', '']
    (out/'README.md').write_text('\n'.join(lines))
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'bindscope')
    parser.add_argument('--data-root', type=Path, default=DATA, help='Directory containing the frozen BindScope source files')
    args = parser.parse_args()
    result = export(args.out, args.data_root)
    print(json.dumps({k:result[k] for k in ('questions','videos','scoring_units','questions_by_type','source_review_counts')},ensure_ascii=False,indent=2))
