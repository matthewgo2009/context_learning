## Reward
1. Challenge Reward
    - solver的答案 stronger的model judge  Reward = -1 * （0/1）(correctness reward)
    - Repetition Penalty [https://arxiv.org/pdf/2508.05004]
    - Format check penalty
    - measure the relationship between context and question from challenge model
    - challenge需要对于rubric的定义判断好坏（自己对于不同的问题要给出不同的rubric 对于所给出rubric评分标准）

2. Solver Reward
    - stronger model（gpt） 判定 solver model的答案 根据rubric 0/1
    - 鼓励去从context去找到答案（防止从pre train来说/胡说八道）
    - 大模型调用tool去查看context（postive）怎么检测 并 metric
    
/Users/liuyicheng/Project_File/llm_research
judge challenge solver model
prompt的定义(modify r zero)


opt algo: GRPO


