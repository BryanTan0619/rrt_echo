# BindScope QA Release Files

This directory contains the v1.5 QA export used by the current ECHO full-dataset evaluation. Distribution of these files does not imply that all source annotations have completed human review.

- `bindscope_qa.json`: a standard JSON array, with one record per question.
- `bindscope_qa.jsonl`: the same records in JSON Lines format.
- `release_manifest.json`: dataset statistics, source hashes, validation results, and the evaluation protocol.

The export contains **77 videos, 2,806 questions, and 1,557 scoring units**: 1,110 R1–R7 pairs (2,220 questions), 139 R0 pairs (278 questions), and 308 single questions.
By format, there are 2146 multiple-choice questions and 660 true/false questions. Paired units include 984 multiple-choice pairs and 265 true/false pairs. R1–R7 also includes true/false questions.

The earlier count of 2,810 questions assumed 141 R0 pairs. The current R0 source contains 139 pairs, accounting for the four-question difference. No questions were added to reach the earlier count.

| Type | Questions | Paired units | Single units | Total units |
|---|---:|---:|---:|---:|
| R0 | 278 | 139 | 0 | 139 |
| R1 | 190 | 91 | 8 | 99 |
| R2 | 408 | 180 | 48 | 228 |
| R3 | 322 | 140 | 42 | 182 |
| R4 | 440 | 168 | 104 | 272 |
| R5 | 342 | 158 | 26 | 184 |
| R6 | 574 | 266 | 42 | 308 |
| R7 | 252 | 107 | 38 | 145 |

All questions and options are in English. Chinese names use consistent romanization; Chinese on-screen labels are rendered as English translations. This text-only normalization preserves answer keys, question IDs, and pair membership. Previously collected predictions should be identified as using the pre-normalization text.

## Fields and Evaluation

`question_id` is a stable question identifier, `video_id` identifies the video, and `binding_type` takes values R0–R7. For multiple-choice questions, `options` uses keys A–D and `answer` is the correct letter. For true/false questions, option keys and answers are the strings `True` and `False`.

`unit_id` identifies the scoring unit; `unit_type` is `binding_pair`, `presence_pair`, or `single`. Paired questions share a `pair_id` and have `pair_index` 1 or 2; both fields are null for singles. A pair is correct only when both questions are answered correctly. Singles are reported separately and excluded from PairAcc.

R1–R7 PairAcc has a denominator of 1,110; PairAcc including R0 has a denominator of 1,249; All-Unit Accuracy has a denominator of 1,557. Independent uniform guessing gives 6.25% for a pair of four-option questions and 25% for a pair of true/false questions. Model errors across two questions may be correlated, so these baselines do not determine observed text-only scores.

Evaluate each question in a fresh conversation, randomize option positions, and map predictions back to the original options before scoring. Model inputs should contain only the video (or memory allowed by the method), the current question, and its options. Do not provide gold answers, the other question in the pair, `direction`, or scoring metadata.

`direction` preserves the source label; it does not guarantee that every pair contains semantically opposite role queries. The QA files exclude private latent bindings, evidence intervals, source questions, text-only test scores, and reviewer information.

## Remaining Human Review

Although the source filenames contain "reviewed", some records are not marked as human-reviewed. The counts below reflect source metadata and are not a guarantee of final annotation quality:

- `binding_pairs:human_reviewed`: 90
- `binding_pairs:not_marked_human_reviewed`: 1020
- `singles:human_reviewed`: 0
- `singles:not_marked_human_reviewed`: 308

Human review should be completed before these records are presented as a fully validated benchmark. This export checks fields and structural consistency; it does not independently rewatch videos to verify answer correctness.

Reproduce the export with `python tools/export_bindscope_release.py --data-root /path/to/Binding_Dataset --out bindscope`. The exporter preserves the original source files and does not affect running experiments.
