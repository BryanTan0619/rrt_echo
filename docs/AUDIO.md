# 可选音频

默认视频建图传输帧图像；MP4 有音轨不代表 VLM 会听到音轨。ASR 必须显式执行，并以 `--audio-observations` 传入建图流程。

`rrt-echo-audio` 依赖单独准备的 M3-Agent 目录：

```text
m3-agent/
  speakerlab/                         # 可导入的实现
  models/faster-whisper-medium/        # 含 model.bin 的本地 Whisper 模型
  models/pretrained_eres2netv2.ckpt
```

若外部仓库还有额外 Python 依赖，需按该仓库的环境说明安装。本仓库没有复制其代码和权重，`pip install .[audio]` 不会准备这些资源。

流程：源时间戳解码 → Whisper ASR → 声纹窗口与匿名聚类 → 时间对齐文本。输出 `audio.json` 与 `summary.json`。无音轨、无语音、过短或混合说话片段分别保留状态；声纹相似不自动证明与画面人物同一身份。

simple 模式将目标时间范围内 ASR 作为上下文，同时保存语音来源。`rrt-echo-multimodal` 为独立可选研究入口，需额外本地视听模型和人工/自动证据审查；本 base 不宣称自动建立可靠的声纹到人物 registry 绑定。

QA 区分说了什么、谁说的，以及世界中实际发生什么。仅有台词不能证明台词内容为真，匿名声纹簇也不能证明人物姓名。
