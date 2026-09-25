# Code map and reproduction notes

## Code map

| File under `src/alignment_course/` | Responsibility |
|---|---|
| `data_prep.py`, `prepare_hh.py` | 从 shared Assistant marker 分离 prompt/chosen/rejected；按源顺序切分并记录解析数量 |
| `train_sft.py` | chosen completion-only LoRA SFT，prompt 不计入损失 |
| `merge_lora.py` | 将 SFT adapter 合并为后续 DPO 的基础模型 |
| `train_dpo.py` | TRL DPOTrainer + LoRA；可设置 beta、步数和学习率 |
| `train_reward_model.py` | LoRA sequence classification reward model |
| `train_ppo_pilot.py` | 可选 PPO pilot；捕获运行期错误并写状态文件 |
| `evaluate.py` | completion log probability、preference accuracy、KL 与可选 RM generation comparison |
| `model_utils.py` | 模型/adapter 加载、LoRA 和 device dtype 设置 |
| `training_utils.py` | TRL/Transformers 参数签名适配 |
| `io_utils.py`, `plot_metrics.py` | 读写、随机种子、计时、参数量统计和画图 |

默认 LoRA：r=16, alpha=32, dropout=0.05，目标层为 q/k/v/o_proj 和 gate/up/down_proj。SFT 默认学习率 2e-4，DPO 默认 1e-5；batch size 1、梯度累积 8、max sequence length 1024、DPO max prompt length 512、seed 42。这些是代码默认值；历史 JSON 没有记录全部超参数，不应视为完整执行配置快照。

## Entry points

在仓库内进入独立项目目录：

```bash
cd projects/qwen_alignment
bash scripts/setup_env.sh
bash scripts/run_smoke.sh
```

setup 使用 uv 创建 Python 3.11 环境并安装项目。需要安装 uv、可下载模型/数据的网络，以及足够计算资源。依赖范围在 `pyproject.toml`，不是锁定环境；此同步仅做静态语法与内容核对，未安装重型依赖或重跑 GPU 训练。

- `scripts/run_main_experiment.sh`：2k 数据设置，SFT → merge → DPO → RM → PPO pilot → evaluation。
- `scripts/run_dpo_extended.sh`：依赖已生成的小规模 SFT merged model 与 RM，运行 beta=0.1、600 步。
- `scripts/run_large10k_autodl.sh`：10k 数据设置，8k/1k/1k，SFT 500 步与 DPO 1,000 步；通过 BASE_MODEL 与 WORK_DIR 覆盖云服务器路径。
- `scripts/sft_baseline.py`：单独计算 SFT reference 的 preference accuracy。

## Known reproduction gaps

1. `evaluate.py` 即使不做生成评价也会加载 reward model；large10k 脚本依赖另行准备的 `artifacts/main/reward_adapter/final`。直接运行 large10k 脚本并不是完全独立的一键复现。
2. 模型权重与数据未上传；先执行数据准备/训练，或自行提供对应模型与 adapter。`results/` 是只读历史证据，运行输出写入 `artifacts/`。
3. 训练和评价源码保持原样，未在本次整理中修复依赖兼容性或改变模型逻辑。TRL 私有可选集成开关、版本宽泛依赖与平台差异可能影响重跑。
4. `prepare_hh.py` 要求解析后样本量达到分割总数；无效源记录可能导致提前报错。源数据版本未固定。
5. `configs/smoke.json` 是配置资料；当前 shell 入口直接传参，并不自动读取此 JSON。
6. 小规模 beta=0.5 的结果已归档，但未找到专门的 shell 入口；可通过 `train_dpo.py --beta 0.5 --max-steps 200` 结合现有路径参数运行。
7. 脚本设置本地代理/镜像的方式属于原始运行环境配置；请按自己的环境调整。

## Archive provenance

37 个源文件和证据文件由本地 `llm_alignment_course` 整理而来，准确列表见 `SOURCE_MANIFEST.json`。小规模最终 CUDA 结果来自 `artifacts/llm_small1600_results.tar.gz`，扩展结果来自 `artifacts/llm_large10k_results.tar.gz`，提取为便于浏览器检索的 JSON。早期 MPS 结果与旧版“待填”报告未混入主表。README、CV_BRIEF、RESULTS 与本页为本次新写的审阅说明。
