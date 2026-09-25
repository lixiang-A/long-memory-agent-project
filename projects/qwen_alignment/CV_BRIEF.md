# CV brief / 网页版阅读材料

## Recommended project title

**Parameter-Efficient Preference Alignment of Qwen3-0.6B with SFT and DPO**

项目性质：课程研究项目。具体起止日期、导师、团队规模、个人分工和答辩状态需由本人补充；代码与结果文件不能单独证明个人独立贡献或论文发表。

## English CV bullets

- Implemented a LoRA-based SFT-to-DPO pipeline for Qwen3-0.6B using Anthropic HH-RLHF, including preference-pair parsing, completion-only supervised fine-tuning, adapter merging, and held-out evaluation.
- Evaluated DPO across two dataset sizes (1,600 and 8,000 training pairs), with small-scale comparisons of beta (0.1/0.5) and training duration (200/600 steps); analyzed preference accuracy, log-probability margins, and divergence from the SFT reference.
- Observed held-out preference accuracy of 59.8% for DPO versus 59.1% for SFT on 1,000 test pairs, while documenting small-data null results and the need for multi-seed validation.

如篇幅有限，保留前两条即可。上面的 implemented/evaluated 以用户实际负责这些工作为前提；团队项目应调整为 contributed to 并写明负责模块。

## 中文简历素材

- 基于 Qwen3-0.6B 与 HH-RLHF 实现 LoRA 后训练流程，覆盖偏好对解析、仅回答部分计算损失的 SFT、权重合并、DPO 及留出集评价。
- 对比 1,600 与 8,000 条训练偏好对设置，并在小规模设置下考察 beta=0.1/0.5 与 200/600 步训练，联合分析偏好准确率、log-probability margin 和相对 SFT 的 KL。
- 扩展设置下，DPO 在 1,000 个测试偏好对上的准确率为 59.8%，SFT 为 59.1%；保留小数据下无稳定增益的结果，并明确统计验证局限。

## Claim → evidence

| Claim | Evidence | Boundary |
|---|---|---|
| 实现 SFT / DPO 流程 | `src/alignment_course/train_sft.py`, `train_dpo.py`, `merge_lora.py` | 使用已有框架，不声称提出 SFT/DPO 或从头训练基础模型 |
| LoRA 参数高效微调 | `model_utils.py`, `results/large10k/dpo_beta0.1_steps1000/run_summary.json` | 约 1.665% 可训练参数；不是实测显存/成本降低比例 |
| beta 与训练步数对照 | `results/small1600/` | 3 个 DPO 设置，不是完整网格搜索；未完成 LoRA rank 消融 |
| 59.1% → 59.8% | `results/large10k/eval_dpo_beta0.1_steps1000.json` | +0.7 个百分点，不是 +7%；离线排序准确率，不是生成胜率 |
| reward model 训练 | `results/small1600/reward_adapter/run_summary.json` | loss 约 0.662 不证明 reward model 的泛化质量 |
| PPO pilot 实现 | `src/alignment_course/train_ppo_pilot.py` | 没有完整 PPO 成功结果；RM win rate 被跳过 |

## 尚未证实或仍属后续计划

多 seed、显著性检验、独立人工/LLM judge 生成评价、拒答率与安全性评估、错误案例分类、LoRA 配置消融、完整 RLHF/PPO 对比、发表或录用。不要把早期计划或论文讨论改写为已完成实验。

## 可复制到网页版的请求

请读取本仓库 `projects/qwen_alignment/README.md`、`CV_BRIEF.md`、`RESULTS.md`、`REPRODUCIBILITY.md`，并按需核对 `src/alignment_course/` 与 `results/`。基于可追溯事实为我润色英文 academic CV 中的 Qwen 项目，输出项目标题和 2–3 条简洁 bullets。将它与根目录 long-memory agent 项目分开；保留课程研究尺度，不夸大 0.7 个百分点的观察性差异，不把 PPO pilot、后续计划或生成评价写成完成成果。涉及日期、个人分工、导师或发表状态时请询问我，不自行推断。
