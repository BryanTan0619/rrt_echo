# Development

```bash
python -m pip install -e '.[dev,perception]'
ruff check src tests tools
ruff format --check src tests tools
pytest -q
python -m build
```

保持图核心可在无 GPU 环境中运行。身份、角色、时间、日志或检索语义变更应使用反例测试验证。测试通过不代表 VLM 对视频的理解正确。

不要加入数据集专用姓名/答案规则、机器绝对路径、令牌、模型或实验产物。源观察和媒体引用保持可追溯；身份变化通过修订投影表达。README 使用当前主入口，实验性能力与已验证结果分开说明。

Git 提交应排除 `outputs/`、`local_archive/`、数据和凭据。当前仓库是内部研究 base，未指定开源许可证。
