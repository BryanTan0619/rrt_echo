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
    lines = ['# BindScope QA 发布文件', '',
             '本目录导出与当前 ECHO 全量实验一致的 v1.5 问答源；准备文件不代表已完成全部人工审核或已公开发布。', '',
             '- `bindscope_qa.json`：标准 JSON 数组，一条记录对应一道题。',
             '- `bindscope_qa.jsonl`：相同记录的逐行 JSON 版本。',
             '- `release_manifest.json`：统计、源文件哈希、验证结果及评分协议。', '',
             '共 **77 个视频、2,806 道题、1,557 个计分单位**。其中 R1–R7 为 1,110 对 / 2,220 题，R0 为 139 对 / 278 题，single 为 308 题。',
             f'按格式统计：四选一 {question_formats["multiple_choice"]} 题，判断题 {question_formats["true_false"]} 题；配对单位中四选一 {pair_formats["multiple_choice"]} 对、判断题 {pair_formats["true_false"]} 对。R1–R7 源文件同样含判断题，并非全部四选一。', '',
             '旧的 2,810 题统计使用 141 对 R0；当前源文件仅有 139 对，因此少 4 题。没有为了凑数补题。', '',
             '| 类型 | 题数 | 配对单位 | 单题单位 | 全部单位 |',
             '|---|---:|---:|---:|---:|']
    for typ in sorted(questions_by_type):
        lines.append(f'| {typ} | {questions_by_type[typ]} | {paired_by_type[typ]} | {singles_by_type[typ]} | {units_by_type[typ]} |')
    lines += ['', '## 字段与评分', '',
              '`question_id` 是稳定题号，`video_id` 定位视频，`binding_type` 使用 R0–R7。MCQ 的 `options` 为 A–D，`answer` 为对应字母；判断题的选项键与答案为字符串 `True` / `False`。', '',
              '`unit_id` 是计分单位；`unit_type` 为 `binding_pair`、`presence_pair` 或 `single`。配对题共享 `pair_id`，`pair_index` 为 1 或 2；single 的这两项为 null。两题均正确才算一个 pair 正确。单题另报，不混入 PairAcc。', '',
              'R1–R7 PairAcc 分母 1,110；含 R0 的 PairAcc 分母 1,249；All-Unit Accuracy 分母 1,557。随机基线对独立均匀四选一 pair 是 6.25%，对独立均匀判断题 pair 是 25%；实际模型两题错误可能相关，不能据此推导其盲测分数。', '',
              '每道题独立开启对话，随机化选项位置，并将预测映射回原选项后评分。模型只接收视频（或方法允许的 memory）、本题文本与选项；不要把正确答案、另一道 paired question、direction 或计分元数据作为模型输入。', '',
              '`direction` 原样保留源标识，不据此保证每一对都是语义上的正反向角色查询。发布文件没有私有 latent binding、证据时间段、source question、盲测得分或审核员信息。', '',
              '## 剩余审核状态', '',
              '源文件虽命名 reviewed，但包含未标记为 human-reviewed 的记录。以下按源标记统计，不能当作最终质量保证：', '']
    for k,v in sorted(reviews.items()):lines.append(f'- `{k}`：{v}')
    lines += ['', '实际公开发布前仍应完成这些记录的人工审核。本次仅做字段导出与结构核验，没有重新观看视频证明答案正确。', '',
              '可用 `python tools/export_bindscope_release.py --data-root /path/to/Binding_Dataset --out bindscope` 复现导出。不会改动原始数据或正在进行的实验。', '']
    (out/'README.md').write_text('\n'.join(lines))
    return manifest


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'bindscope')
    parser.add_argument('--data-root', type=Path, default=DATA, help='Directory containing the frozen BindScope source files')
    args = parser.parse_args()
    result = export(args.out, args.data_root)
    print(json.dumps({k:result[k] for k in ('questions','videos','scoring_units','questions_by_type','source_review_counts')},ensure_ascii=False,indent=2))
