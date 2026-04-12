# 如何运行 CL-bench RL

本文说明从环境准备到各类训练/试跑命令的**完整运行方式**。默认基座模型为 **`Qwen/Qwen2.5-3B-Instruct`**（可在命令行或配置中覆盖）。

---

## 1. 环境要求

| 项目 | 说明 |
|------|------|
| Python | 建议 3.10+ |
| PyTorch | 2.x；训练/推理建议有 **NVIDIA GPU** 与足够显存 |
| 网络 | 首次运行需从 Hugging Face 下载数据集 `tencent/CL-bench` 与模型权重 |
| Hugging Face | 若数据集或模型需登录，先执行 `huggingface-cli login` |
| Judge（可选但推荐） | 论文设定为**冻结评测 LLM**；默认通过 **OpenAI API** 调用（如 `gpt-4o`）。需配置 `OPENAI_API_KEY`，否则奖励中的 Judge 项会**退化为启发式** |

---

## 2. 安装依赖

在**仓库根目录**（含 `clbench_rl/`、`scripts/` 的目录）执行：

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

论文附录中与 **BLEU + 层次聚类重复惩罚**相关的功能，建议安装：

```bash
pip install nltk scikit-learn
```

未安装时，重复惩罚等会退化为近似实现。

---

## 3. 环境变量（重要）

### 3.1 OpenAI Judge（与 `reward.use_llm_judge=True` 配合）

训练脚本默认希望使用 **LLM Judge**。任选其一即可：

```bash
export OPENAI_API_KEY="sk-..."     # 推荐：写入 shell 配置，勿提交到 Git
```

或在运行 `scripts/train_adversarial.py` 时传入：

```bash
python scripts/train_adversarial.py --openai-api-key "sk-..."
```

### 3.2 其他常用变量

```bash
export HF_HOME=~/.cache/huggingface   # 可选：缓存目录
export TOKENIZERS_PARALLELISM=false  # 多进程时减少 tokenizer 警告
```

启动脚本 `scripts/launch_8gpu.sh` 内还会设置 `NCCL_*` 等分布式相关变量。

---

## 4. 运行方式概览

| 目的 | 入口 | 说明 |
|------|------|------|
| 不加载大模型，只测数据与奖励 | `run_pipeline.py --dry-run` | 最快自检 |
| Rollout + 指标（`ReinforceTrainer`） | `run_pipeline.py` | 小样本试跑流水线 |
| **论文主流程：Solver + Challenger 双 GRPO** | `train_adversarial.py` | 对抗自博弈训练 |

下面分节给出具体命令。

---

## 5. 快速验证（不加载本地大模型）

只拉取少量数据并跑奖励逻辑，用于确认数据集与依赖正常：

```bash
python scripts/run_pipeline.py --dry-run --max-samples 5
```

---

## 6. 流水线试跑：`run_pipeline.py`

加载 **同一 Hugging Face 模型** 作为 Challenger 与 Solver，做环境一步、生成与奖励统计（适合小样本调试）：

```bash
python scripts/run_pipeline.py --max-samples 10 --model Qwen/Qwen2.5-3B-Instruct
```

常用参数：

| 参数 | 含义 |
|------|------|
| `--max-samples` | 最多处理样本数 |
| `--model` | 双端共用的模型名 |
| `--checkpoint-dir` | checkpoint 目录 |
| `--dry-run` | 不加载模型，仅测数据与奖励 |

---

## 7. 对抗训练（主流程）：`train_adversarial.py`

对应 **Self-Evolving ICL / 非对称对抗**设定下的 **Solver + Challenger 双端 GRPO**。未传参时，多数超参与 `clbench_rl/config/default_config.py` 中 `get_default_config()` 一致；命令行会覆盖其中对应项。

### 7.1 单机单进程试跑（建议先限制样本）

```bash
export OPENAI_API_KEY="sk-..."   # 若要用真实 Judge

python scripts/train_adversarial.py \
  --max-samples 32 \
  --epochs 1 \
  --checkpoint-dir checkpoints/adversarial_smoke
```

### 7.2 单机多卡：`torchrun`

```bash
torchrun --nproc_per_node=8 scripts/train_adversarial.py \
  --max-samples 100 \
  --epochs 1
```

（实际是否多卡数据并行取决于当前 `AdversarialTrainer` 是否接入 DDP；多进程前请先确认显存与进程数。）

### 7.3 DeepSpeed 启动脚本（8 GPU）

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/launch_8gpu.sh
```

可通过环境变量改默认模型与超参，例如：

```bash
export MODEL=Qwen/Qwen2.5-7B-Instruct
export EPOCHS=2
bash scripts/launch_8gpu.sh -- --max-samples 200
```

`launch_8gpu.sh` 末尾的 `"$@"` 会传给 `train_adversarial.py`。

### 7.4 Hugging Face Accelerate + DeepSpeed

```bash
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_adversarial.py
```

DeepSpeed 配置见 `configs/ds_config_zero2.json`、`configs/ds_config_zero3.json`。

### 7.5 `train_adversarial.py` 常用参数

| 参数 | 含义 |
|------|------|
| `--model` | Challenger / Solver 共用基座（默认 3B） |
| `--solver-model` / `--challenger-model` | 分别覆盖两端模型 |
| `--epochs` | 训练轮数 |
| `--lr` | Solver 学习率 |
| `--challenger-lr` | Challenger 学习率 |
| `--group-size` | GRPO 组大小 G |
| `--kl-beta` / `--clip-eps` | KL 与裁剪系数 |
| `--max-samples` | 限制样本数；不传则按配置使用全量或默认 |
| `--data-split` | 数据集 split，如 `train` |
| `--checkpoint-dir` | checkpoint 保存目录 |
| `--save-every` / `--log-every` / `--ref-sync-every` | 保存、日志、参考模型同步步频 |
| `--no-llm-judge` | 关闭 API Judge，仅用启发式 |
| `--judge-model` | Judge 模型名（默认 `gpt-4o`） |
| `--output-dir` | 训练结束后写入 `final_metrics.json` 的目录 |

---

## 8. 仅用 Python 调用 Trainer（高级）

在任意工作目录，将仓库根目录加入 `PYTHONPATH` 或先 `pip install -e .`（若已配置），然后：

```python
from clbench_rl.config.default_config import merge_config
from clbench_rl.trainer.grpo_trainer import GRPOTrainer
from clbench_rl.trainer.adversarial_trainer import AdversarialTrainer

cfg = merge_config({
    "data": {"split": "train", "max_samples": 32},
    "training": {"epochs": 1, "checkpoint_dir": "checkpoints"},
    "grpo": {"group_size": 4},
})
# GRPOTrainer(cfg).train()           # 仅 Solver 侧 GRPO 示例
# AdversarialTrainer(cfg).train()    # 双端对抗 GRPO
```

默认配置键包括：`data`、`challenge_model`、`solver_model`、`reward`、`training`、`grpo`。合并规则见 `merge_config()`（顶层键浅合并，子字典合并覆盖）。

---

## 9. 输出位置

- **Checkpoint**：由 `--checkpoint-dir` / 配置中 `training.checkpoint_dir` 决定（注意 `.gitignore` 可能忽略 `checkpoints/`）。
- **对抗训练指标**：`train_adversarial.py` 在 `--output-dir`（默认 `outputs/`）下写入 **`final_metrics.json`**。

---

## 10. 常见问题

1. **首次运行很慢**  
   需下载 CL-bench 与 Qwen 权重，属正常现象。

2. **Judge 一直走启发式**  
   检查是否设置 `OPENAI_API_KEY`，或是否加了 `--no-llm-judge`。

3. **显存不足**  
   减小 `--max-samples`、`grpo.group_size`，或换更小模型（保持 3B），或使用 ZeRO 配置 / 更小 batch（若你在代码中调整）。

4. **多卡行为**  
   若未配置分布式，多进程可能各自加载完整模型；调试时建议先用**单进程** `--max-samples` 小规模跑通。

---

## 11. 相关文件

| 路径 | 内容 |
|------|------|
| `clbench_rl/config/default_config.py` | 默认超参与模型名 |
| `scripts/train_adversarial.py` | 对抗训练 CLI |
| `scripts/run_pipeline.py` | 流水线 CLI |
| `scripts/launch_8gpu.sh` / `scripts/launch_torchrun.sh` | 多卡启动示例 |
| `configs/accelerate_config.yaml` | Accelerate + DeepSpeed |
| `README.md` | 项目简介与其它说明 |

更细节的实现与论文公式对应关系见代码内注释及仓库中的方法说明 PDF（若有）。
