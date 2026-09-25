# Results and evidence

本页基于历史 CUDA 聚合结果 JSON 核对；不是本次重新训练或重测的结果。JSON 来源见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。

| Setting | Test pairs | SFT accuracy | DPO accuracy | DPO mean margin | DPO approx KL | Evidence |
|---|---:|---:|---:|---:|---:|---|
| 1,600 train; beta 0.1; 200 steps | 200 | 55.0% | 55.0% | 22.9702 | 0.035427 | [JSON](results/small1600/eval_dpo_200_fixed_v2.json) |
| 1,600 train; beta 0.1; 600 steps | 200 | 55.0% | 54.5% | 25.0273 | 0.152306 | [JSON](results/small1600/eval_dpo_600_fixed_v2.json) |
| 1,600 train; beta 0.5; 200 steps | 200 | 55.0% | 55.0% | 22.4403 | 0.012376 | [JSON](results/small1600/eval_dpo_beta05_fixed_v2.json) |
| 8,000 train; beta 0.1; 1,000 steps | 1,000 | 59.1% | 59.8% | 29.2280 | 0.128154 | [JSON](results/large10k/eval_dpo_beta0.1_steps1000.json) |

SFT mean margin：小规模 22.0173，扩展规模 26.2075。扩展设置准确率差为 0.7 个百分点，即净多判对 7 个偏好对；不能据此恢复逐样本的改正/退化数量。

## Training evidence

| Run | Steps | Train loss | Recorded training seconds | Evidence |
|---|---:|---:|---:|---|
| Small SFT | 120 | 2.325245 | 341.33 | [JSON](results/small1600/sft_adapter/run_summary.json) |
| Small DPO beta 0.1 | 200 | 0.667590 | 641.85 | [JSON](results/small1600/dpo_adapter/run_summary.json) |
| Small DPO beta 0.1 | 600 | 0.561191 | 1873.48 | [JSON](results/small1600/dpo_600steps/run_summary.json) |
| Small DPO beta 0.5 | 200 | 0.656447 | 685.56 | [JSON](results/small1600/dpo_beta0.5/run_summary.json) |
| Small reward model | 200 | 0.662079 | 553.44 | [JSON](results/small1600/reward_adapter/run_summary.json) |
| Large SFT | 500 | 2.238795 | 1399.72 | [JSON](results/large10k/sft_adapter/run_summary.json) |
| Large DPO beta 0.1 | 1,000 | 0.621207 | 3279.78 | [JSON](results/large10k/dpo_beta0.1_steps1000/run_summary.json) |

Runtime 使用 `train_runtime_seconds`，不代表端到端耗时或费用。不同方法的训练 loss 不能直接比较。

## Metric definitions and limitations

- Preference accuracy：同一 prompt 下 chosen completion 的 token log probability 之和大于 rejected 时记为正确，平局记为错误。它受长度、截断与概率校准影响，不是人类偏好生成胜率。
- Mean margin：上述 chosen 与 rejected 序列 log-probability 总和之差的样本均值。
- Approx KL：在数据中 chosen response 的 teacher-forced token 上计算 policy 到 SFT reference 的词表分布 KL，再按回答和样本平均；不是 on-policy 生成轨迹的 KL。
- 所有主表文件均为 `n_generation_prompts=0` 且 `rm_win_rate.status=skipped_no_ppo_model`。不存在可据此报告的 DPO-vs-PPO 生成胜率。
- 两档规模的测试集合不同，且 SFT/DPO 训练步数也变化；不能把跨规模差异单独归因于数据量。没有多 seed、置信区间或配对显著性检验。
- 数据从 HH-RLHF 的 train split 前缀确定性切分，项目留出测试集不是官方 test split。未记录数据/模型的固定 revision；本地源代码快照与历史云端执行版本没有 commit 绑定。
