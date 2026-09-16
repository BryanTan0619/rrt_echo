# 架构与数据契约

发布入口：`rrt-echo-offline --profile simple`、`rrt-echo-qa`。核心不是一份 caption，而是可追溯的联合事实与绑定解释。

## 写入：RRT

1. 按源视频时间采样，使用镜头/动作过程组织目标片段和邻接上下文。区域绑定必须引用具体 media ID 和框。
2. 每片段一次联合观察：实例、事件的有向角色、属性/文字、观察时间、局部对应及缺口。动作观察与全局身份未决可以同时存在。
3. 分开表达短时连续 `continues`、同一身份 `same_identity` 与部位归属 `part_of`。模型输出的 ownership 显式区分 part/whole。
4. 有限预算的对应比较，轮转人物身份、归属与物体身份候选；已处理 pair 可在补充轮排除，避免重复调用。

simple 模式不使用 QA、标准答案或人工 caption 建图，不启用全视频故事索引、多轮感知修复或额外视听验证。

## 存储：ECHO

`RRTMemory` 的 canonical source 为 `ledger/observations.jsonl`。记录含追加观察、绑定提案/撤销、音频和文字修订，带日志完整性校验。`facts.jsonl`、`bindings.jsonl`、实体注册表和 `hypergraph.json` 均为派生导出，不是多个可独立修改的事实源。

| 层 | 关键字段 | 语义 |
|---|---|---|
| 证据 | media ID、URI、时间、hash、region box | 指向源媒体；引用合法不代表模型理解正确 |
| 实例 | instance ID、kind、description、regions | 当前局部可见对象，不直接等于全局人物 |
| 事实 | kind、predicate、roles、value、joint_evidence、evidence_by_slot、observed_times | 同一观察的动作/属性、角色和证据 |
| 绑定 | relation、source、target、verdict、evidence IDs、status | 可修订身份/归属解释 |
| 派生视图 | resolved_roles、owner_projections、dependencies、unresolved_slots | 通过已接受关系解释局部端点 |
| 快照 | snapshot_id、binding revision、acceptance policy | reader 固定读取的版本 |

`continues` 可提供身份连续依据，不意味着状态持续；身份相同不意味着事件可以合并。共同观察的文字和动作只有共享端点、共享帧才组成属性束，且不自动认定动作产生该文字。说话内容保存为报告内容，不自动提升为世界事实。

## 绑定验收

验收依次检查端点、类型、引用和冲突。复用同一人物框且无独立锚点时，相关角色/绑定保持未决。两个完整人物承担字面 carry/hold 的不同角色时，身份解释不得把它们折叠成同一实体；传递路径形成冲突的候选簇保持 disputed。

这些是条件一致性约束，不是视频真值证明。对错误局部事实应用约束可能拒绝正确对应；应报告未决覆盖与 QA，而不是追求更少人物簇。

## 检索与阅读

`rrt.reader.payload_for` 在底层 `retrieve` 上增加问题时间范围、选项假设检索、属性绑定视图、身份/归属路径与多模态证据闭包。Reader 收到整个事实及其必要引用，不从不同时间/角色的词片段拼接新事实。

选项文字只用于召回候选，不写入图。未决实体 ID 不同不代表真实人物不同；证据缺口不是反证。预算不足会记录省略/闭包状态，不得把残缺路径宣称为完整支持。

图内支持审计与源视频语义验收分开。当前 CLI 提供诊断 QA，完整媒体验收仍需外部流程。

## 内部兼容边界

顶层 `cli.py`、`pipeline.py` 等保留公共基础设施与合成回放的兼容实现；`echo_perception` 是适配层的历史包名。`rrt-echo build/qa` 为旧接口，不作为本 base 的建图/QA 主入口。高级 `legacy` profile、multimodal/Omni 代码为可选研究路径，默认不执行。
