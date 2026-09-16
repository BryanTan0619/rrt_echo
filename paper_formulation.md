# ICLR2026 论文逻辑梳理

> 本文档整合三份笔记(`Formulation：抑制多模态长视频理解幻觉.md`、`Idea整理.md`、`总体论文框架整理.pdf`)并经多轮讨论重构而成。
> 主线已升级为「统一 binding 视角」。**实验部分待后续讨论,本文仅留占位。**

---

## 定位(一句话)

> 我们第一次站在统一视角审视长视频理解的**全部**幻觉,指出无论 out-video(编造没发生的)还是 in-video(错配已发生的),本质都是**多模态信息的 binding 错误**;据此提出用 **temporal hypergraph** 在理解视频时把所有多模态信息正确 binding,**统一、免任务训练、适配任意模型**地抑制所有幻觉,并给出配套的完整分类与诊断数据集。

---

## 一、Introduction 逻辑

### 1.1 短视频 vs 长视频(为什么长视频值得重新审视)
短视频:实体/事件少、切换少、绑定不需长程 → 语义易正确提取。
长视频新增 challenge:细粒度证据多且短时快现、事件级语义关系复杂(电影级叙事)、物体状态/跨事件长程联系需显式保存 → 幻觉更高发、种类更多。

### 1.2 幻觉分类:第一次做完整审视
以往工作各自处理某一类(event、推理、entity…)。我们不设限,完整审视并分成两支:
- **Out-Video**:无中生有,编造没出现过的人/物/动作/事件;
- **In-Video**:把出现过的人/物/动作/事件/特征**错配**(多子类)。

并强调 in-video **更难、更易被忽视、长视频里最高发**——是框架内的主攻硬核。

### 1.3 统一洞见:万物皆 binding
两支的共性:都是 binding 错误。
- **Out** = binding 到**根本不存在**的对象;
- **In** = binding 到**错误的**对象。

配 **颗粒度契约(granularity contract)**守住 thesis:
> hypergraph 承诺一个表示颗粒度(可细可粗,可只保留主要人物/事件);低于该颗粒度的内容本就不进图,被约束在图上作答的模型只会**受控省略(omission)**,绝不编造(fabrication)。因此幻觉(=无效 binding 的断言)在颗粒度契约内被彻底消除,sub-granularity 细节仅是覆盖率问题、非正确性问题。

### 1.4 Research Gap
现有做法(外置显式 memory + RAG / 内部 latent KV cache)都没在「binding 结构」层面建模,无法统一处理各类幻觉;且社区缺少一个覆盖全类型的分类与诊断工具。

### 1.5 Contributions(四条)
1. **统一视角 + 完整分类**:首次把长视频**全部**幻觉(out+in)统一为 binding 错误并给出完整 taxonomy——更本质、更高维。
2. **binding 形式化**:用 `GlobalSupport / ScopedSupport` 两谓词统一定义 out(绑到不存在)与 in(绑错对象)。
3. **诊断型数据集**:覆盖 in+out 全类型的长视频幻觉 benchmark,让任意模型/memory 方法自测**每类幻觉 bias** 并针对性提升(含 injection 压力测试子集)。
4. **方法**:temporal hypergraph,**correct-by-construction**、**task-level training-free**、model-agnostic;construction 质量作为后续可持续改进的开放轴。

---

## 二、统一分类树(Taxonomy —— Method 与 Dataset 的桥梁)

```
Binding Error(根)
├── Binding-to-Nothing  = Out-Video(图里无对应超边 → 编造)
│     ├── Entity/Object Fabrication
│     ├── Action Fabrication
│     └── Event Fabrication
└── Binding-to-Wrong    = In-Video(元素对、绑定错)
      ├── Event Membership   R1 Predicate–Event Leakage / R2 Participant–Event Leakage
      ├── Relation Role      R3 Role Permutation / R6 Attribute–Owner Leakage
      ├── Entity Identity    R4 Entity Continuity Failure
      └── Temporal Binding   R5 State–Version Leakage / R7 Event–Relation Corruption
```

### In-Video 七类失效模式(R1–R7)明细

| 大类 | 编号 | 类型 | Binding 错误 | 例子 |
|---|---|---|---|---|
| Event Membership | R1 | Predicate–Event Leakage | predicate ↔ event | A 在 Event B 拿杯子,答成在 Event A |
| Event Membership | R2 | Participant–Event Leakage | entity ↔ event | Event A 是 P1 搬箱子,答成 P2 |
| Relation Role | R3 | Role Permutation | entity ↔ role | P1 给 P2,答成 P2 给 P1 |
| Entity Identity | R4 | Entity Continuity Failure | observation ↔ identity | 后面相似人物被当成前面的人 |
| Temporal Binding | R5 | State–Version Leakage | entity ↔ state ↔ time | 换衣前红色,答成换衣后蓝色 |
| Relation Role | R6 | Attribute–Owner Leakage | attribute ↔ entity | P1 穿黄 P2 穿蓝,答成 P2 穿黄 |
| Temporal Binding | R7 | Event–Relation Corruption | event_i ↔ event_j | 两次开门被混合 / 先后答反 |

---

## 三、Problem Formulation(统一 binding)

**Pipeline**:`V → (RRT) → H(temporal hypergraph) → 按 c_q 读取 scoped 子图 → Y`,作答**被约束在图上**。

**统一定义(问题层)**:查询 scope `c_q = (entity, event, role, time, state)`;答案 `a` 正确 ⇔ `GlobalSupport(a,V)=1` 且 `ScopedSupport(a,V,c_q)=1`。
- **Out-Video**:`GlobalSupport(a,V)=0`(内容不在视频里)。
- **In-Video**:`GlobalSupport(a,V)=1` 且 `ScopedSupport(a,V,c_q)=0`(在视频里但绑错 scope)。

**机制(correct-by-construction,不做证据检索核验)**:H 中存在某超边 ⇔ 对应内容真实发生且其参与者/角色/时间/状态被正确绑定;作答只读取正确 scope 的子图。
- **Out 被挡**:没发生 → 无超边 → 读不出 → 不能断言;
- **In 被挡**:正确 n-ary 超边 + scoped 读取 → 返回正确绑定。

> 全文杠杆收敛为一句:**图建对,答案就对**。`Global/Scoped` 仅作问题定义,不作运行时检索/核验步骤。

**三阶段 SAH(诊断维度,服务 RQ2,非主 formulation)**:in-video 误绑在 memory 中的发生位置——

| 阶段 | 核心机制 | 典型表现 |
|---|---|---|
| Write-time SAH | entity resolution / event / role / 属性归属错误;不确定 binding 过早提交 | 一开始就把动作写到错误人物或事件 |
| Consolidation SAH | merge / compression / summary / state update 破坏原 binding | 两人被合并、重复事件被压成一个、旧状态被覆盖 |
| Retrieval SAH | scope 解析错误、相似度检索跨事件污染、role/state/time 过滤缺失 | 存对了,但取出另一个事件的事实 |

> 用途:逐阶段定位不同 memory 方法的幻觉发生位置与类型偏向(RQ2 的洞见贡献)。

---

## 四、Method:Temporal Hypergraph(ECHO)

### 4.1 RRT(Recognition–Registration–Tracing)
多模态抽取(Visual: VLM + 人脸检测/识别;Audio: ASR + 声纹 + 声学证据)。ObservationPacket 只存局部轨迹/证据/候选分布,**不擅自宣称跨段身份**;全局身份由**离线 identity graph** 看完整段视频后判定 → 解决跨场景身份连续性(换装/换背景不产生新 entity,也不误合并相似人物),契合 offline formulation。

### 4.2 Evidence Hypergraph Memory
四类超边:Event / Identity / State+Temporal / Event-relation。天然表示 **n-ary relation**,只有联合信息充分时才 commit;**versioned facts**,新观察触发依赖修订 → 防合并/压缩破坏绑定。

### 4.3 Scoped Read-off
按 `c_q` 读取正确 scope 的子图(而非带核验的检索)→ 防跨事件/跨角色/跨状态污染。

### 4.4 训练与泛化
hypergraph 本身**无任务层训练**;仅子模块(人脸/声纹/ASR)使用现成模型,与任务正交、可替换 → 保证 **model-agnostic** 与泛化性。construction 质量是本框架下的**开放改进轴**,后续工作可在子模块精度、颗粒度自适应等方向继续提升。

---

## 五、Dataset(诊断型 benchmark)

### 5.1 覆盖范围
in + out 全类型(见第二节 taxonomy),**每类独立报点**,暴露方法的分类偏向(如「entity 追踪强但 role 交换弱」这类被总分掩盖的偏差)。

### 5.2 两个组件
1. **taxonomy-labeled 评测集**:测每类幻觉 bias;
2. **injection stress test**:向目标长视频插入不相关片段,测 robustness(机器难 robust,人类可轻松分开,如视频里的广告)。

### 5.3 Pipeline
1. 长视频 VLM dense caption 生成 + 前置 audit;
2. 原子关系互换,构造 counter-factual QA 选项(含不同 level);
3. 生成候选问题与 QA 选项;
4. Human audit + refinement。

### 5.4 数据源
YouTube 视频 + 现有 benchmark(ELV-Halluc、M3 Robots、LV-Bench)。

> **命名待定**:数据集总名(含 out+in、taxonomy 标注)可能需比 EventSplice 更宽的名(如 HalluBind / LV-HalluBench);EventSplice 专指其中的 injection 子集。待定。

---

## 六、Related Works 站位

- **Hallucination benchmarks**(ELV-Halluc / OmniHalluc-L / NOAH …):各自审视子集或只提概念;我们**统一 out+in 为单一 binding taxonomy,并给出原子失效模式 + 方法**。
- **Long Video Memory**(M3-Agent / WorldMM / MAGIC-Video / ReflectWorldMM / Video-EM / EventCausalRAG …):只记录「正确 perception」,结构上不区分事件归属;我们用 temporal hypergraph 显式做正确 binding。
- **Multimodal-RAG / HypergraphRAG**:借 n-ary 表达但非为长视频时序绑定设计;我们把 temporal validity + 正确身份绑定作为一等公民。

---

## 七、Experiments

> **占位——待后续讨论。**
> 已知锚点:RQ2(逐阶段定位不同方法的幻觉位置/偏向)从「防守性举证」升级为「诊断型洞见贡献」;实验需证明统一方法在各类幻觉上带来一致、可观的下降(概念统一 + 经验收益即成立,不以「构建完美」为门槛)。

---

## 附:决策台账(已锁定)

1. 主线 = 统一 binding;in-video 为框架内主攻硬核。
2. 颗粒度契约守 thesis,sub-granularity → 受控省略(覆盖率问题,非正确性问题)。
3. 机制 = correct-by-construction,删除 grounding / evidence 检索核验话术。
4. task-level training-free,子模块可训练但正交;删除一切「可学习

绑定目标 / loss-reward lifecycle」。
5. 三阶段(Write/Consolidation/Retrieval)保留为**诊断维度**(RQ2),非主 formulation。
6. 「图建对→答案对」定位为 framework 贡献 + 开放改进轴,不作自设负担。
7. Contributions 共四条(含新增的诊断型数据集)。
8. 数据集命名待定;实验部分待后续讨论。
