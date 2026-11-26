# REPOSITORY_MANIFEST — claude-GAIA

本地 Claude Code + 模型中转站执行 GAIA 工作负载的代码。

- Repository: `claude-GAIA`
- Local path: `/home/claude-GAIA`
- Python: 3.14（基线）
- Workload: GAIA 2023 validation

## 纳入版本管理的文件

| 文件 | 用途 |
|---|---|
| `run_gaia_claude.py` | GAIA 任务读取、Claude CLI 启动、Bubblewrap 隔离、时间线、答案校验 |
| `plot.py` | 结果分析与绘图（numpy/matplotlib） |
| `requirements.txt` | 运行依赖 |
| `env.example` | 模型中转站环境变量模板（无值） |
| `README.md`, `REPOSITORY_MANIFEST.md`, `.gitignore` | 文档与排除规则 |

## 明确排除

```text
DATASET/              GAIA 数据集
results_claude_gaia/  实验结果
.venv/                虚拟环境
*.pcapng, *.jsonl     原始采集与时间线
.env                  模型中转凭据
__pycache__/          缓存
```

## 与其它仓库的关系

本仓库是 Claude 侧的 GAIA workload 执行器，与 `codex_qwen` /
`codex_qwen_GAIA`（Codex 侧）并列，用于跨 Agent/模型的流量对照；
不共享运行时，也不依赖 `waitung-inference-server-measurement` 的代码。
