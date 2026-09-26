# BindScope QA 发布文件

本目录导出与当前 ECHO 全量实验一致的 v1.5 问答源；准备文件不代表已完成全部人工审核或已公开发布。

- `bindscope_qa.json`：标准 JSON 数组，一条记录对应一道题。
- `bindscope_qa.jsonl`：相同记录的逐行 JSON 版本。
- `release_manifest.json`：统计、源文件哈希、验证结果及评分协议。

共 **77 个视频、2,806 道题、1,557 个计分单位**。其中 R1–R7 为 1,110 对 / 2,220 题，R0 为 139 对 / 278 题，single 为 308 题。
按格式统计：四选一 2146 题，判断题 660 题；配对单位中四选一 984 对、判断题 265 对。R1–R7 源文件同样含判断题，并非全部四选一。

旧的 2,810 题统计使用 141 对 R0；当前源文件仅有 139 对，因此少 4 题。没有为了凑数补题。

| 类型 | 题数 | 配对单位 | 单题单位 | 全部单位 |
|---|---:|---:|---:|---:|
| R0 | 278 | 139 | 0 | 139 |
| R1 | 190 | 91 | 8 | 99 |
| R2 | 408 | 180 | 48 | 228 |
| R3 | 322 | 140 | 42 | 182 |
| R4 | 440 | 168 | 104 | 272 |
| R5 | 342 | 158 | 26 | 184 |
| R6 | 574 | 266 | 42 | 308 |
| R7 | 252 | 107 | 38 | 145 |

## 字段与评分

`question_id` 是稳定题号，`video_id` 定位视频，`binding_type` 使用 R0–R7。MCQ 的 `options` 为 A–D，`answer` 为对应字母；判断题的选项键与答案为字符串 `True` / `False`。

`unit_id` 是计分单位；`unit_type` 为 `binding_pair`、`presence_pair` 或 `single`。配对题共享 `pair_id`，`pair_index` 为 1 或 2；single 的这两项为 null。两题均正确才算一个 pair 正确。单题另报，不混入 PairAcc。

R1–R7 PairAcc 分母 1,110；含 R0 的 PairAcc 分母 1,249；All-Unit Accuracy 分母 1,557。随机基线对独立均匀四选一 pair 是 6.25%，对独立均匀判断题 pair 是 25%；实际模型两题错误可能相关，不能据此推导其盲测分数。

每道题独立开启对话，随机化选项位置，并将预测映射回原选项后评分。模型只接收视频（或方法允许的 memory）、本题文本与选项；不要把正确答案、另一道 paired question、direction 或计分元数据作为模型输入。

`direction` 原样保留源标识，不据此保证每一对都是语义上的正反向角色查询。发布文件没有私有 latent binding、证据时间段、source question、盲测得分或审核员信息。

## 剩余审核状态

源文件虽命名 reviewed，但包含未标记为 human-reviewed 的记录。以下按源标记统计，不能当作最终质量保证：

- `binding_pairs:human_reviewed`：90
- `binding_pairs:not_marked_human_reviewed`：1020
- `singles:human_reviewed`：0
- `singles:not_marked_human_reviewed`：308

实际公开发布前仍应完成这些记录的人工审核。本次仅做字段导出与结构核验，没有重新观看视频证明答案正确。

可用 `python tools/export_bindscope_release.py --data-root /path/to/Binding_Dataset --out bindscope` 复现导出。不会改动原始数据或正在进行的实验。
