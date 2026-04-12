# CL-bench RL

基于 [tencent/CL-bench](https://huggingface.co/datasets/tencent/CL-bench) 的上下文学习实验框架：Challenge / Solver 双模型、多分量 Rubric 奖励，并支持 **GRPO（Group Relative Policy Optimization）** 对 Solver 做策略梯度更新。

**如何安装与运行（命令、环境变量、多卡启动）见 [`RUN.md`](RUN.md)。**

## 环境要求

- **Python** 3.10+（建议）
- **PyTorch** 2.x（GPU 可选，用于本地模型推理与 GRPO）
- 能访问 **Hugging Face**以下载 `tencent/CL-bench`（首次运行需联网）

可选依赖（用于论文附录 B.4 的 BLEU + 层次聚类重复惩罚）：

```bash
pip install nltk scikit-learn
```

未安装时会自动退化为基于 n-gram / 词重叠的近似实现。

## 安装

```bash
cd llm_research   # 或你的仓库根目录
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 快速验证（不加载模型）

仅拉取少量数据并跑启发式奖励逻辑，用于检查环境与数据集：

```bash
python scripts/run_pipeline.py --dry-run --max-samples 5
```

## 运行完整流水线（ReinforceTrainer）

会加载本地 Hugging Face 模型（默认 `Qwen/Qwen2.5-3B-Instruct`），对每条样本执行：环境一步 → Solver 生成 → 计算 Solver / Challenge 奖励。**注意**：该路径主要做 rollout 与指标统计；是否保存 checkpoint 由配置中的 `training.save_every` 等决定，与完整 REINFORCE 梯度更新不同。

```bash
# 小样本试跑（建议先限制 max_samples）
python scripts/run_pipeline.py --max-samples 10 --model Qwen/Qwen2.5-3B-Instruct
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--max-samples` | 最多处理多少条数据 |
| `--model` | Challenge / Solver 共用的 HuggingFace 模型名 |
| `--checkpoint-dir` | checkpoint 目录 |
| `--dry-run` | 不加载模型，只测数据与奖励 |

## GRPO 训练（Solver）

在项目根目录下，用 Python 调用 `GRPOTrainer`（需 GPU 与足够显存加载模型；可按需减小 `data.max_samples` 与 `grpo.group_size`）：

```python
from clbench_rl.config.default_config import merge_config
from clbench_rl.trainer.grpo_trainer import GRPOTrainer

cfg = merge_config({
    "data": {"split": "train", "max_samples": 32},
    "training": {"epochs": 1, "lr": 1e-5, "checkpoint_dir": "checkpoints"},
    "grpo": {"group_size": 4, "clip_eps": 0.2, "kl_beta": 0.04},
})
trainer = GRPOTrainer(config=cfg)
trainer.train()
```

GRPO 相关默认项见 `clbench_rl/config/default_config.py` 中的 `grpo` 段。

## 配置说明

默认配置由 `clbench_rl/config/default_config.py` 的 `get_default_config()` 提供，可通过 `merge_config({...})` 覆盖顶层键（`data`、`challenge_model`、`solver_model`、`reward`、`training`、`grpo`）。

奖励模块为 `clbench_rl/rewards/rubrics_reward.py` 中的 `RubricsReward`，支持与「自演化 ICL / 对抗出题」思路对齐的多项分量（对抗项、重复惩罚、格式、上下文相关性、Rubric–证据对齐等）。Challenger 侧权重可使用 `w1_adversarial` … `w5_rubric`；配置里若仍使用旧字段名（如 `challenge_correctness_weight`），训练器会映射到对应新参数。

## 使用 LLM Judge（OpenAI）

将 `reward.use_llm_judge` 设为 `true`，并为 `RubricsReward` 传入 `api_client`（例如 OpenAI SDK 的 client）。需在环境中配置 `OPENAI_API_KEY`。  
未开启时，Judge 相关项使用启发式或规则近似，便于离线调试。

示例（在自定义脚本中）：

```python
from openai import OpenAI
from clbench_rl.rewards.rubrics_reward import RubricsReward

reward_fn = RubricsReward(
    use_llm_judge=True,
    judge_model="gpt-4o",
    api_client=OpenAI(),
)
```

训练器在 `use_llm_judge=True` 时会通过 `OPENAI_API_KEY` 自动构建 Judge 的 `api_client`；自定义脚本可如上手动传入 `OpenAI()`。

## 仓库结构（简要）

```
clbench_rl/
  config/          # 默认与合并配置
  data/            # CL-bench 加载
  env/             # CLBenchEnv：一步交互
  models/          # Challenge / Solver 封装
  rewards/         # RubricsReward 等
  trainer/         # ReinforceTrainer, GRPOTrainer
scripts/
  run_pipeline.py      # 流水线入口
  train_adversarial.py   # 对抗双 GRPO 训练
RUN.md                 # 运行说明
```

## 许可证与引用

使用数据集与第三方模型时，请遵守其各自许可协议。若发表相关工作，请引用 CL-bench 与所使用模型的官方说明。
