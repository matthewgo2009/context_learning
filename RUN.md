# CL-bench RL 运行指南（多卡版本）

**本文档以单机多 GPU 正式训练为主路径**（推荐 **8×GPU**，通过 DeepSpeed / `torchrun` 启动）。单卡、小样本仅用于**冒烟测试与调试**；生产实验请按下方「多卡对抗训练」一节执行。

- **默认基座模型**：`Qwen/Qwen3-4B-Instruct-2507`（可覆盖）。Qwen3 需 **transformers≥4.51**，见 `requirements.txt`。
- **Judge**：冻结评测 LLM，默认 **`gpt-4o-mini`**（OpenAI API）。密钥放在 **`.env`** 的 `OPENAI_API_KEY=`或环境变量中（`.env` 已 gitignore，勿提交仓库）。

---

## 1. 环境要求

| 项目 | 说明 |
|------|------|
| Python | 建议 3.10+ |
| GPU | **多卡训练**：单机多卡 NVIDIA GPU（脚本默认8 卡）；需安装对应 CUDA 的 PyTorch |
| 网络 | 首次需从 Hugging Face 拉取 `tencent/CL-bench` 与模型权重 |
| Hugging Face | 需登录时：`huggingface-cli login` |
| OpenAI | Judge 调用需有效 API Key；无 Key 时 Judge 退化为启发式 |

---

## 2. 安装（仓库根目录）

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

可选（论文附录 BLEU + 聚类重复惩罚更完整）：

```bash
pip install nltk scikit-learn
```

---

## 3. 多卡训练前必配

### 3.1 Judge API（`.env` 推荐）

在仓库根目录创建 `.env`（勿提交 Git）：

```bash
OPENAI_API_KEY=sk-...
```

`scripts/launch_8gpu.sh` 会自动 `source .env`；直接跑 Python 时，`build_judge_api_client()` 也会尝试加载根目录 `.env`。

或临时导出：

```bash
export OPENAI_API_KEY="sk-..."
```

也可：`python scripts/train_adversarial.py --openai-api-key "sk-..."`

### 3.2 分布式 / 稳定性

```bash
export TOKENIZERS_PARALLELISM=false
# 可选
export HF_HOME=~/.cache/huggingface
```

`launch_8gpu.sh` 内已设置 `NCCL_*` 等，可按集群情况调整。

---

## 4. 多卡主流程：对抗训练（论文对齐）

**入口**：`scripts/train_adversarial.py`（Solver + Challenger 双端 GRPO）。

### 4.1 推荐：DeepSpeed 8 卡（`launch_8gpu.sh`）

```bash
bash scripts/launch_8gpu.sh
```

默认：`deepspeed --num_gpus=8` + 环境变量中的 `MODEL`、`EPOCHS`、`SOLVER_LR`、`CHALLENGER_LR` 等。覆盖示例：

```bash
export MODEL=Qwen/Qwen2.5-7B-Instruct
export EPOCHS=2
bash scripts/launch_8gpu.sh -- --max-samples 200
```

末尾 `"$@"` 会原样传给 `train_adversarial.py`。

### 4.2 备选：`torchrun`（无 DeepSpeed）

```bash
bash scripts/launch_torchrun.sh -- --epochs 1 --max-samples 100
```

`NUM_GPUS` 默认 8，可按机器修改脚本或：

```bash
NUM_GPUS=4 bash scripts/launch_torchrun.sh -- --max-samples 50
```

### 4.3备选：Accelerate + DeepSpeed

```bash
accelerate launch --config_file configs/accelerate_config.yaml scripts/train_adversarial.py
```

ZeRO 配置：`configs/ds_config_zero2.json` / `ds_config_zero3.json`。

### 4.4 多卡注意事项

当前 `AdversarialTrainer` 的 DDP/数据并行与 DeepSpeed 的完整集成以你本地环境为准；若多进程各加载一份完整模型，请注意显存。不确定时先用 **单进程小样本**（下一节）验证逻辑，再上多卡。

---

## 5. 单卡 / 单进程（仅调试用）

不用于正式多卡实验，仅快速验证数据、Judge、保存路径：

```bash
python scripts/train_adversarial.py \
  --max-samples 32 \
  --epochs 1 \
  --checkpoint-dir checkpoints/adversarial_smoke
```

---

## 6. `train_adversarial.py` 常用参数

| 参数 | 含义 |
|------|------|
| `--model` | Challenger / Solver 共用基座 |
| `--solver-model` / `--challenger-model` | 分别指定两端模型 |
| `--epochs` | 训练轮数 |
| `--lr` / `--challenger-lr` | Solver / Challenger 学习率 |
| `--group-size` | GRPO 组大小 G |
| `--kl-beta` / `--clip-eps` | KL、裁剪 |
| `--max-samples` | 限制样本数 |
| `--data-split` | 如 `train` |
| `--checkpoint-dir` | checkpoint 目录 |
| `--save-every` / `--log-every` / `--ref-sync-every` | 保存、日志、参考模型同步 |
| `--no-llm-judge` | 关闭 API Judge |
| `--judge-model` | 默认 `gpt-4o-mini` |
| `--output-dir` | 训练结束写入 `final_metrics.json`（默认 `outputs/`） |

默认配置合并自 `clbench_rl/config/default_config.py`。

---

## 7. 训练曲线与最终结果可视化

训练过程会往 **`{checkpoint_dir}/metrics.jsonl`** 追加指标（见各 Trainer 实现）。

生成图：

```bash
python scripts/plot_metrics.py checkpoints/metrics.jsonl --out figures/
python scripts/plot_metrics.py outputs/final_metrics.json --out figures/
```

多实验对比：

```bash
python scripts/plot_metrics.py run_a/metrics.jsonl run_b/metrics.jsonl --labels "a" "b" --out figures/
```

---

## 8. 快速自检（不加载大模型）

```bash
python scripts/run_pipeline.py --dry-run --max-samples 5
```

---

## 9. 流水线试跑：`run_pipeline.py`（ReinforceTrainer）

小样本 rollout + 指标，非对抗双 GRPO 主路径：

```bash
python scripts/run_pipeline.py --max-samples 10 --model Qwen/Qwen3-4B-Instruct-2507
```

| 参数 | 含义 |

|------|------|
| `--max-samples` | 最多样本数 |
| `--model` | 双端共用模型名 |
| `--checkpoint-dir` | checkpoint |
| `--dry-run` | 不加载模型 |

---

## 10. Python 直接调用 Trainer（高级）

```python
from clbench_rl.config.default_config import merge_config
from clbench_rl.trainer.adversarial_trainer import AdversarialTrainer

cfg = merge_config({
    "data": {"split": "train", "max_samples": 32},
    "training": {"epochs": 1, "checkpoint_dir": "checkpoints"},
    "grpo": {"group_size": 4},
})
AdversarialTrainer(config=cfg).train()
```

---

## 11. 输出位置

| 内容 | 位置 |
|------|------|
| Checkpoint | `--checkpoint-dir` / `training.checkpoint_dir` |
| 逐步指标 | `{checkpoint_dir}/metrics.jsonl` |
| 训练结束汇总 | `outputs/final_metrics.json`（`--output-dir` 可改） |
| 图表 | `scripts/plot_metrics.py` 的 `--out`（默认 `figures/`） |

---

## 12. 常见问题

1. **首次很慢**：下载数据集与模型权重正常。  
2. **Judge 变启发式**：检查 `OPENAI_API_KEY` 与 `.env`。  
3. **显存不够**：减小 `--max-samples`、`group_size`，或换更小模型、开 ZeRO。  
4. **多卡异常**：先单进程小样本跑通，再开多卡；检查 `CUDA_VISIBLE_DEVICES`、端口 `MASTER_PORT` 是否冲突。

---

## 13. 相关文件

| 路径 | 说明 |
|------|------|
| `scripts/launch_8gpu.sh` | **多卡主入口**（DeepSpeed 8 GPU） |
| `scripts/launch_torchrun.sh` | 多卡 `torchrun` |
| `scripts/train_adversarial.py` | 对抗训练 CLI |
| `scripts/plot_metrics.py` | 指标可视化 |
| `configs/accelerate_config.yaml` | Accelerate + DeepSpeed |
| `clbench_rl/config/default_config.py` | 默认超参 |

更多项目背景见 `README.md`。
