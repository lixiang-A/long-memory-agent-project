# Qwen3-0.6B Preference Alignment: SFT and DPO

> 整理日期：2026-09-25。独立课程研究项目，存放于此仓库供网页版读取与 CV 润色；与根目录的 long-memory agent 原型是两个不同项目。

## 阅读入口

- [CV 素材与事实边界](CV_BRIEF.md)：中英文简历 bullet、可用与不可用表述。
- [实验结果与证据](RESULTS.md)：逐项链接原始 JSON。
- [代码与复现说明](REPRODUCIBILITY.md)：模块、配置、命令与已知限制。
- [来源清单](SOURCE_MANIFEST.json)：代码与聚合结果的原始相对路径及 SHA-256。

## 项目概述

基于 Qwen/Qwen3-0.6B 和 Anthropic/hh-rlhf 偏好数据，构建 LoRA SFT → 合并 SFT adapter → LoRA DPO → held-out 偏好对评价的研究流程。研究训练步数、DPO beta 与数据规模下的结果变化，并实现 reward-model 训练与可选 PPO pilot。

**已归档结果**：小规模 1,600/200/200，扩展规模 8,000/1,000/1,000（train/validation/test）。扩展设置的离线 preference accuracy 从 SFT 的 59.1% 到 DPO 的 59.8%，增加 0.7 个百分点。小规模 DPO 无稳定 accuracy 增益。没有多 seed 或显著性检验，不能声称显著改善、SOTA、生成质量提升或完整 RLHF 成功。

代码使用 Python、PyTorch、Transformers、TRL、PEFT、Datasets。LoRA 默认 rank=16、alpha=32、dropout=0.05；归档策略模型可训练参数为 10,092,544，占含 adapter 总参数约 1.665%。本地开发使用 MPS，主表采用 CUDA 归档；项目论文记载云端设备为 RTX 4090 24GB，JSON 本身仅记录 CUDA。

本次同步保留原始训练/评价代码，新增说明文档与结果索引，没有重新训练。未上传模型权重、优化器状态、环境缓存、原始 HH 对话、私人申请资料或论文封面。
