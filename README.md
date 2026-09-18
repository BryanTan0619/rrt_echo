# RRT–ECHO

从长视频构建可追溯的事件记忆，再使用有绑定约束的检索完成 QA。

这是当前 **base 实现**：默认 `simple` 流程。RRT 提取局部事件与实体对应，ECHO 保存联合事实、证据和可修订的身份解释。回答与证据支持分开计分；建图成功不等于视频语义验收通过。

## 1. 方法与代码

```text
MP4 ──→ 帧采样与连续片段 ──→ RRT 联合观察 ──→ 有限预算实体对应
         ↑ 可选 ASR 文本                         ↓
                                  ECHO 观察日志与绑定验收
                                               ↓
                                      hypergraph.json
                                               ↓
                         scoped retrieval → QA → 独立的图内支持审计
```

| 模块 | 职责 | 主要代码（`src/rrt_echo/`） |
|---|---|---|
| RRT | 每片段联合提取人物/物体/区域、事件角色、文字、时间和源帧；比较 `continues`、`same_identity`、`part_of` | `rrt/pipeline.py`、`rrt/observation.py`、`rrt/association.py` |
| ECHO | 追加原始观察；校验归属和身份冲突；生成超图与实体注册表 | `rrt/memory.py`、`rrt/hypergraph.py`、`echo_perception/graph.py` |
| Retrieval | 按问题和时间检索完整事实，补齐身份、归属与证据依赖 | `retrieval.py`、`rrt/binding_scope.py`、`rrt/reader.py:payload_for` |
| QA | 只读图，不回看视频；强制选择答案，再单独审计支持 | `rrt/reader.py`、`evaluation.py` |
| 可选音频 | 时间戳 ASR 与匿名声纹簇；保留未决说话人 | `rrt/audio.py`、`rrt/multimodal.py` |

`schema.py`、`memory.py`、`identity.py`、`storage.py` 是公共图存储基础；`echo_perception/` 与 `perception/` 保留主链路依赖的采样、校验和模型适配器。内部仍有兼容实现，使用下面的主入口即可。历史实验和旧报告不随本仓库发布。

### 超图保存什么

局部事实的角色端点与全局身份解释分开。例如一个写字事件保存 `writer=p1, instrument=pen1, surface=arm1`；文字观察保存 `owner=arm1, value="18 km"`，另用 `part_of(arm1,p1)` 解释身体归属。跨片段通过 `same_identity(p1,p7)` 解释为同一实体。

同一事件的角色、时间、区域和证据一起检索；不会仅凭文字共现将事实拼成新事件。文字与动作共享承载物、共享帧时可以形成共同观察的属性束，但这不自动证明文字由该次动作写成。身份修订不覆盖原始局部端点。

## 2. 安装

Python 3.11+，在仓库根目录执行：

```bash
git clone https://github.com/BryanTan0619/rrt_echo.git
cd rrt_echo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,perception]'
pytest -q
```

图存储与检索核心无第三方运行依赖；视频流程使用 PyAV、Pillow、NumPy 和 JSON Schema。VLM 服务单独部署，本仓库不包含模型权重、视频或 QA 数据。

## 3. 从 MP4 建图

准备支持图片输入和 JSON 输出的 OpenAI-compatible VLM 服务。示例显式使用图片模式；不要求服务端安装本项目的视频传输扩展。

```bash
export RRT_VLM_BASE_URL=http://127.0.0.1:8001/v1
export RRT_VLM_MODEL=your-vision-model
# 如服务需要鉴权：export RRT_VLM_API_KEY=...

rrt-echo-offline \
  --video /path/to/video.mp4 \
  --source-video-id video01 \
  --out outputs/video01 \
  --profile simple --mode images \
  --model "$RRT_VLM_MODEL" --base-url "$RRT_VLM_BASE_URL" \
  --sample-fps 2 --local-seconds 12 \
  --processor-max-pixels 401408 \
  --max-local-calls 100 --max-identity-batches 24 \
  --association-workers 2 --max-tokens 4096 --timeout 120 --budget 3600
```

输出目录必须不存在。`source-video-id` 要与 QA 数据中的 `video_id` 一致。预算与调用上限可能导致部分覆盖；查看 `report.json`、`coverage.json`、`failures.json`，不要把 `DONE.json` 当作质量通过证明。超时预算不是对远端模型服务的硬终止保证。

默认 `simple` 不启用全局故事索引、多轮局部修复或额外说话人视听验证；仍进行局部联合观察和有限预算对应比较。`--profile legacy` 保留部分多阶段实验能力，不是本 base 的使用路径。

```text
outputs/video01/
  ledger/observations.jsonl   # canonical：追加观察、绑定/撤销、音频与修订记录
  hypergraph.json             # QA 使用的完整快照，带 snapshot_id
  entity_registry.json       # 可重建的实体注册表；孤立实例保持未决
  storyline.json             # 按时间排列的事实视图，不是新生成的故事
  facts.jsonl, bindings.jsonl # 派生导出，不是另一套独立事实源
  view_manifest.json
  coverage.json, report.json, failures.json, request_inventory.json
  ...                        # 源帧、请求/响应和诊断产物
```

## 4. 检查检索，再运行 QA

先导出与 QA 相同的 scoped retrieval payload，不调用模型：

```bash
python tools/inspect_memory.py \
  --graph outputs/video01/hypergraph.json \
  --question 'Who is carrying the child near the end?' \
  --out outputs/video01_inspect
```

检索使用词项加权、问题时间范围及完整事实闭包，不依赖向量数据库。选择题的各选项可作为分别检索的假设，不能成为事实。结果包含事件角色、身份链、归属链、证据 ID 和缺口；不足时保留未决。

准备 JSONL，每行一道题：

```json
{"question_id":"q1","video_id":"video01","hallucination_type":"R2","question":"Who carries the child?","options":{"A":"The adult in the plaid shirt","B":"The woman in white"},"ans":"A"}
```

上例只说明格式，需使用自己视频的题目与标注。支持 `options` 字典、题干中的 `A) ...` 选项，以及 `question_type="True/False"`。`hallucination_type` 用于分组统计，不控制建图；`ans` 只用于评分，不送给模型。

```bash
rrt-echo-qa \
  --graph outputs/video01/hypergraph.json \
  --questions /path/to/questions.jsonl \
  --acceptance outputs/video01_inspect/acceptance.json \
  --diagnostic --out outputs/video01_qa \
  --model "$RRT_VLM_MODEL" --base-url "$RRT_VLM_BASE_URL" \
  --workers 2
```

检查工具生成的 acceptance 明确为 `passed=false`，所以这里使用 `--diagnostic`。只有完成对应快照的视频验收后，才能提供真实的 `passed=true` 记录；不要为了运行 QA 改成通过。

每题通常两次模型调用：一次回答，一次图内审计。Reader 只收到检索图的文本表示与证据引用，不收到图像、原始视频、caption 或标准答案。

- `summary.json`：准确率、执行失败、按 R 类型统计、图内支持分布。
- `results.json`：逐题答案、实际 reader payload、断言审计与缺口。
- `calls.json` / `question_calls/`：调用耗时和 token 使用。
- `manifest.json`：冻结快照、模型、访问范围和诊断模式。

强制作答时，即使证据不足也会选择一个选项。`correct_with_support` 在没有独立媒体审计时保持 `null`，不能用 QA 准确率替代绑定验收。

## 5. 重建、可视化与无 GPU 示例

```bash
# 从日志重建，不重新调用模型；原始日志仍保存在源目录
python tools/rebuild_simple_memory.py \
  --ledger outputs/video01/ledger --out outputs/video01_rebuilt

# 文字、承载物与引用源帧的 HTML；源帧需要可访问
python tools/render_text_evidence.py \
  --graph outputs/video01/hypergraph.json --out outputs/video01_text
python -m http.server 8767 --directory outputs

# 不使用视频/模型的合成结构示例
rrt-echo replay examples/observations.jsonl \
  --identity examples/identity.jsonl --out outputs/toy
rrt-echo inspect outputs/toy/hypergraph.json --question 'Who attacks whom?'
```

跨机器移动源帧后，可给文字可视化工具加 `--path-map /old/prefix=/new/prefix`。合成示例只验证结构，`synthetic://` 不是实际视频证据；`rrt-echo inspect` 是底层检索示例，当前 QA 完整 payload 请使用 `tools/inspect_memory.py`。

## 6. 可选 ASR

已有 M3-Agent 代码和本地 Whisper/ERes2NetV2 权重时：

```bash
python -m pip install -e '.[audio]'
rrt-echo-audio --video /path/to/video.mp4 \
  --m3-root /path/to/m3-agent --device cuda --out outputs/video01_audio
```

给建图命令追加 `--audio-observations outputs/video01_audio/audio.json`。详见 [音频说明](docs/AUDIO.md)。视觉模型看到的是时间对齐的 ASR 文本，图片模式不直接听音频。声纹簇是匿名假设，不自动等于可见人物；ASR 内容不自动成为世界事实。

## 7. 当前边界

该 base 已跑通一次完整开发视频：45 个窗口、349 条事实，QA 20/28。不是跨视频泛化指标。OCR、人物区域/类型、跨镜头身份、动作发起角色仍有错误；冲突约束可以阻断错合并，也可能造成身份断链。详见 [版本状态](docs/STATUS.md)。

- [架构与数据契约](docs/ARCHITECTURE.md)
- [QA 与评估协议](docs/EVALUATION.md)
- [开发与检查](CONTRIBUTING.md)

视频、模型、凭据、实验输出和本地历史归档均不进入 Git。当前为内部研究 base，未声明开源许可证。
