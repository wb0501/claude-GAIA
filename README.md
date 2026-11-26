# claude-GAIA

本地 Claude Code（经由模型中转站）运行 GAIA 工作负载的执行器与结果绘图代码。

## 用途

- `run_gaia_claude.py`：读取 GAIA 任务、在 Bubblewrap 隔离环境中启动 Claude
  CLI、收集 Claude 流式事件、记录 Codex/Claude 时间线、校验答案
- `plot.py`：解析 `results_claude_gaia/` 下的结果，生成论文用图表

## 环境

- Python >= 3.11（基线 3.14）
- Claude Code CLI，使用 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` /
  `ANTHROPIC_API_KEY` 指向模型中转站
- `bwrap`、`tshark` / `dumpcap`（如启用抓包）
- `numpy`、`matplotlib`（`plot.py`）

## 快速开始

```bash
python -m pip install -r requirements.txt
cp env.example .env      # 填入中转站地址与 token；.env 不入库
python run_gaia_claude.py --help
python plot.py --help
```

## 数据策略

`DATASET/`、`results_claude_gaia/`、`.venv/`、PCAP/JSONL 日志和
`.env` 全部由 `.gitignore` 排除。
