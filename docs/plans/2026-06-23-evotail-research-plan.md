# EvoTail Research Implementation Plan

**Goal:** 验证并缩小蛋白语言模型在低进化支持度蛋白上的性能差距，形成可复现、可发表的 EvoTail 研究成果。

**Architecture:** 研究分为“现象验证—数据与基准—最小方法—结构扩展—论文复现”五阶段。主方法采用按边际系统发育信息量分配训练预算的 continual pretraining，并用 group-robust 目标改善最弱 evolutionary-support 分组；结构条件不变性仅在主假设成立后加入。

**Tech Stack:** Python 3.11、PyTorch、Hugging Face Transformers、ESM-2、MMseqs2、MAFFT/HMMER、CATH/PDB、ProteinGym、DVC 或 W&B、Hydra、pytest、SLURM/DeepSpeed。

---

## 0. 当前仓库实现状态（2026-06-23）

本仓库目前是“阶段 1：验证进化支持度鸿沟”的研究脚手架，**不是已完成的 EvoTail 模型或已得到结论的实验包**。它已经实现了可在提供外部数据和 GPU 环境后执行的最小管线：

1. `src/evotail/data/support_score.py` 读取蛋白元数据，验证序列，汇总可用的 `nearest_identity`、多层级簇大小、`msa_neff` 与 `taxon_count`。每个可用分量先做百分位排名，再取均值为连续 `evolutionary_support_score`；缺失分量以中位数填补并保留缺失计数。分位组 `0` 表示最低支持度。
2. 同一模块生成有上下限、均值归一的 `tail_sampling_weight`，但尚未存在实际训练数据采样器；该列目前只是后续阶段的输入元数据。
3. `scripts/score_proteingym_assay.py` 与 `src/evotail/scoring/esm_masked_marginal.py` 用冻结的 ESM-2（默认 150M）给 ProteinGym 的替换突变打分。它对每个突变位点在完整突变序列上下文中做 masked-marginal 对数似然比；多突变体取各位点之和，仅支持不改变序列长度的替换突变。
4. `scripts/analyze_support_gap.py` 与 `src/evotail/analysis/support_gap.py` 产生每个 assay 的 Spearman、分层 bootstrap 置信区间、可用混杂变量的 OLS 诊断、JSON 汇总和 Figure 1 草图。Slurm 模板将三步拆成支持度表、按 assay 数组打分、统计分析。
5. `tests/` 覆盖上述纯 Python 核心逻辑，使用合成数据；数据、模型权重与生成的结果不在仓库中。

因此，当前仓库已部分完成 Task 1、Task 2 和 Task 3 的**代码骨架**，但尚未完成这些任务的科学验收：没有真实数据产物、没有已报告的 support gap、没有 650M 对照，也没有可据此发表的结论。Task 1 所说的 35M 训练 smoke test 尚不可执行，因为训练循环尚未实现。

### 当前实现与计划之间必须保留的边界

- 代码中的每 assay Spearman 是项目自己的实现，尚未调用 ProteinGym 官方聚合脚本；在完成评分约定交叉验证前，不能将其称为官方 leaderboard 等价结果。
- `cluster_sequences.py` 只提供可审阅的 MMseqs2 命令构造和结果表解析；原始同源搜索、MSA、分类学覆盖度和输入数据质检仍需按 Task 2 的流程在仓库外执行并留存记录。
- `configs/train.yaml` 仅是未来 continual-pretraining 的占位配置。系统发育信息量采样、GroupDRO/CVaR、token-matched 消融、EvoTail-Bench、泄漏审计、结构条件不变性与最终评测均尚未实现。
- Phase 1 中观察到的相关性仅能支持关联描述，不足以证明训练数据不均衡造成性能差距；是否进入阶段 2 仍由本计划的预注册成功/止损标准决定。

下一项实际工作是准备版本化的外部元数据与 ProteinGym manifest，完成一次冻结 ESM-2 的 Phase 1 运行，并先核对项目评分与 ProteinGym 官方约定的一致性；只有这些前置条件满足后，才应开始训练代码。

---

## 1. 研究问题与预注册承诺

### 核心假设

即使使用 UniRef50 去冗余训练，预训练语料在同源家族、MSA 有效深度和系统发育覆盖上仍不均衡；这种不均衡造成单序列 PLM 在低进化支持度蛋白上的表征和突变效应预测劣势。

### 不应使用的表述

- 不称 ESM-2 “直接记忆了 UniProt 的百万重复序列”。
- 不称仅凭重采样“学习了生物物理第一性原理”。
- 不把“无 Pfam 注释”直接等同于 orphan protein。
- 不用 UMAP/t-SNE 图作为表征有效性的主证据。

### 成功标准

1. 在控制序列长度、assay 类型、突变深度和功能类别后，低支持度与零样本性能下降存在稳定关系。
2. EvoTail 相比相同 token 数、相同骨干模型的随机 continual pretraining，在 tail strata 的主要指标有显著提升。
3. Head 指标下降不超过预先声明的容忍阈值（推荐：平均 Spearman 相对下降不超过 1%，且 bootstrap CI 不支持显著恶化）。
4. 至少在 ProteinGym 之外的一项严格结构或功能泛化任务上复现收益。

### 止损标准

若阶段 1 中效果量接近零或经控制变量后不稳定，则停止“representation collapse”叙事；项目转向“极端环境/特定生物域的 domain adaptation”，不进入大规模 EvoTail 训练。

## 2. 预期仓库与可复现规范

### Task 1: 初始化可复现研究仓库

**Files:**
- Create: `README.md`
- Create: `configs/base.yaml`
- Create: `configs/data.yaml`
- Create: `configs/train.yaml`
- Create: `configs/eval.yaml`
- Create: `src/evotail/__init__.py`
- Create: `data/README.md`
- Create: `results/README.md`

**Step 1:** 固化环境：Python、CUDA、PyTorch、ESM checkpoint、UniRef/Pfam/CATH/ProteinGym 的版本与下载日期。

**Step 2:** 在 `data/README.md` 记录每个数据源的许可证、发布日期、原始路径、预处理命令和任何排除规则。

**Step 3:** 在所有训练配置中记录随机种子、总 token、峰值显存、GPU 小时和 checkpoint SHA。

**Step 4:** 为每次实验生成不可覆盖的运行 ID；训练、数据处理和评测使用相同的运行 ID 关联。

**Acceptance check:** 一台新机器仅根据 README 和 config 即可重建一个 35M 模型的 smoke test。

## 3. 阶段 1：验证“进化支持度鸿沟”

### Task 2: 定义连续 evolutionary-support score

**Files:**
- Create: `src/evotail/data/support_score.py`
- Create: `src/evotail/data/cluster_sequences.py`
- Create: `tests/data/test_support_score.py`
- Output: `artifacts/support/protein_support.parquet`

**Step 1:** 对每个蛋白计算多尺度特征：最近邻序列 identity、UniRef90/50/30 或 MMseqs2 嵌套簇规模、MSA `N_eff`、分类学覆盖度、序列长度与低复杂度比例。

**Step 2:** 用预先声明的公式构建 support score；保存原始分量，不只保存最终分数。推荐先使用分位数分层而非任意绝对阈值。

**Step 3:** 抽样人工核查每个分位组的序列，排除明显片段、非标准氨基酸序列和超长低复杂度污染。

**Step 4:** 单元测试应验证：相同输入得到相同分数；空 MSA 有显式处理；极端簇大小不会产生无穷权重。

### Task 3: 制作关键 Figure 1

**Files:**
- Create: `src/evotail/analysis/support_gap.py`
- Create: `src/evotail/analysis/figures.py`
- Output: `results/phase1/figure1_support_gap.pdf`
- Output: `results/phase1/support_gap_stats.json`

**Step 1:** 在 ProteinGym 上运行冻结的 ESM-2 150M 与 650M（如资源允许）的 zero-shot variant scoring。

**Step 2:** 以每个 DMS assay 为单位，计算官方 Spearman；只用官方聚合脚本产生主结果。

**Step 3:** 进行分层回归和 bootstrap：性能 ~ support score + 长度 + 突变深度 + assay selection type + 功能类别。

**Step 4:** 报告每个 support 分位组的 assay 数、蛋白数、均值、95% CI，禁止只报 tail 最好看的一个阈值。

**Acceptance check:** 得到一张可证伪的图和完整统计表；它决定是否进入阶段 2。

## 4. 阶段 2：建立匹配型长尾基准

### Task 4: 构建 EvoTail-Bench

**Files:**
- Create: `src/evotail/benchmarks/match_support_pairs.py`
- Create: `src/evotail/benchmarks/splits.py`
- Create: `docs/benchmark-card.md`
- Output: `artifacts/benchmarks/evotail_bench_v1.parquet`

**Step 1:** 从 CATH/SCOPe、ProteinGym、EC/GO 数据中构造评测对象；训练、验证、测试按序列 identity 和结构层级严格隔离。

**Step 2:** 在同一 fold/superfamily 或相近功能组内匹配 head/tail 样本，匹配长度、可观测性和任务类型；保存每个匹配的理由。

**Step 3:** 定义 Evolutionary Support Gap (ESG)：同一任务、相同结构/功能层级下，head 与 tail 的平均性能差。

**Step 4:** 对每个基准发布 data card：构建日期、潜在泄漏、分组样本数、已知偏差和推荐指标。

**Acceptance check:** 独立基线可复现 ESG；分组之间不存在明显的序列同源泄漏。

## 5. 阶段 3：最小 EvoTail 方法

### Task 5: 实现系统发育信息量采样器

**Files:**
- Create: `src/evotail/data/phylo_weight.py`
- Create: `src/evotail/data/samplers.py`
- Create: `tests/data/test_samplers.py`
- Modify: `configs/data.yaml`

**Step 1:** 用层级同源图估计每条序列的边际进化信息贡献；若不能可靠重建全树，明确采用“hierarchical homology diversity”近似，而不声称完整系统发育树。

**Step 2:** 实现温度采样 `p_i ∝ score_i^τ`，并加入最小/最大概率裁剪。随机采样、簇均匀采样和逆簇大小采样必须保留作对照。

**Step 3:** 用固定 100 万 token 的模拟采样，验证新采样器提高多尺度簇覆盖，而不会只偏向短序列或低质量单例。

**Step 4:** 单元测试总概率为 1、固定 seed 可复现、裁剪生效、空分组报错清晰。

### Task 6: 实现 group-robust continual MLM

**Files:**
- Create: `src/evotail/training/group_dro.py`
- Create: `src/evotail/training/train_mlm.py`
- Create: `tests/training/test_group_dro.py`
- Modify: `configs/train.yaml`

**Step 1:** 将 support quantile 作为 group，而非将氨基酸 token 当作长尾类别。

**Step 2:** 实现 GroupDRO 或 CVaR objective；训练初期采用 warm-up，随后平滑提升最弱组权重；权重必须裁剪。

**Step 3:** 加入 head-retention monitor：持续记录 head/overall MLM loss、校准指标和各组有效采样量。

**Step 4:** 确保采样和 loss 加权可单独开关，避免默认双重上权尾部。

**Acceptance check:** 在 35M smoke run 上各组损失、梯度范数和权重都稳定，无 NaN、无单组独占 batch。

### Task 7: 运行受控消融

**Experiments:**
1. 原始 ESM-2 checkpoint（不继续训练）
2. 随机 continual MLM（token-matched）
3. 簇均匀 / 逆频率采样（token-matched）
4. 系统发育信息量采样
5. 随机采样 + GroupDRO
6. EvoTail = 信息量采样 + GroupDRO

**Step 1:** 先在 ESM-2 35M 或 150M 上固定数据快照、token 数、学习率搜索预算和随机种子运行全部消融。

**Step 2:** 每个设置至少 3 个 seeds；若预算不够，完整方法与随机 continual 至少 3 seeds，其余 1 seed 标记为 exploratory。

**Step 3:** 统一报告总 token、GPU 小时、参数量、推理成本，禁止把更多训练数据或更多 token 当作方法收益。

**Acceptance check:** EvoTail 在 Phase 1 的 tail 指标上超过 token-matched continual baseline，并满足 head-retention 约束。

## 6. 阶段 4：可选的结构条件不变性扩展

### Task 8: 仅在主方法有效后加入 EvoTail-Struct

**Files:**
- Create: `src/evotail/data/structural_pairs.py`
- Create: `src/evotail/training/structural_invariance.py`
- Create: `tests/training/test_structural_invariance.py`
- Modify: `configs/train.yaml`

**Step 1:** 使用实验 PDB/CATH 结构构造低序列 identity、同结构层级、support 不同的蛋白对；对任何预测结构标记来源与生成日期。

**Step 2:** 构建条件对比目标：同结构/功能邻域内减少 support-induced representation gap，而不是无条件把所有 tail 蛋白拉向 head。

**Step 3:** 控制伪负样本：相同超家族但未被标为正对的蛋白不能直接视为负样本。

**Step 4:** 将此模块明确写为弱结构监督；分别报告 sequence-only EvoTail 和 EvoTail-Struct，不能混淆贡献来源。

**Stop rule:** 若结构模块仅提高有结构标签任务、却不提高 sequence-only ProteinGym/remote-homology，则不作为论文主张。

## 7. 阶段 5：最终评测与稳健性检查

### Task 9: 评估矩阵

**Primary:** ProteinGym zero-shot DMS substitutions；按官方规则报告 Spearman、NDCG、AUC、MCC，并按 MSA depth / support strata / assay 类型分层。

**Secondary:** CATH/SCOPe remote homology 或 fold-level holdout；报告 retrieval Recall@k、macro-F1 和结构层级分层结果。

**Tertiary:** 长尾 EC 分类或极端环境蛋白性质预测；必须采用 sequence-identity + taxonomy-aware split，报告 macro-F1、balanced accuracy 与 few-shot 曲线。

**Robustness:**
- 时间切分或训练数据排除实验，检查测试序列及近邻是否进入 continual corpus；
- 训练 token、模型大小、采样温度、GroupDRO 系数的敏感性；
- 每个主结果提供 bootstrap CI 与 per-assay/per-family 散点；
- 与 ESM-2、ESM-1v、随机 continual、强 ProteinGym 单序列基线比较；如可获得，再与 ESM-C/ESM3 等现代模型明确区分参数量与训练数据差异。

### Task 10: 结果判定表

| 结果模式 | 结论与动作 |
|---|---|
| Tail 提升，head 持平，至少两类任务复现 | 完整主线成立，准备顶会版本 |
| Tail 提升，head 小幅下降 | 调整为 Pareto 改善，分析权衡，收紧 retention 约束 |
| 仅 ProteinGym 提升 | 检查 likelihood calibration、数据泄漏与 benchmark overfit；不主张泛化 |
| 仅结构模块提升 | 将工作定位为 weakly supervised structure-aware adaptation |
| 阶段 1 无 support gap | 停止 EvoTail 主线，转向特定领域 adaptation |

## 8. 时间与算力安排

| 周数 | 交付物 | 推荐资源 |
|---|---|---|
| 1–2 | support score、Figure 1、止损决策 | 1–4 GPU 或现成 checkpoint 推理 |
| 3–4 | EvoTail-Bench、数据卡、泄漏审计 | CPU + 少量 GPU |
| 5–6 | 35M/150M sampler 与 GroupDRO 消融 | 4–8 A100 等效资源 |
| 7–9 | 三 seed 主实验、ProteinGym + remote homology | 8–16 A100 等效资源 |
| 10–11 | 结构扩展或 robustness 实验 | 视阶段 3 结果决定 |
| 12 | 图表、论文初稿、代码清理与复现包 | 少量 GPU |

先不要承诺“原始训练量的 5%–10%”或“48 张 A100 数周”。以实际 token 数、吞吐量和 GPU 小时估算；150M 的完整可行性验证应先于 650M 训练。只有 150M 获得稳定效应，才升级至 650M，并保持同一训练 token 比例。

## 9. 论文结构与投稿门槛

### 论文的单句贡献

“EvoTail 通过按边际进化信息而非观测序列频率分配预训练预算，缩小蛋白语言模型在低 evolutionary-support 蛋白上的性能鸿沟，同时保持高支持度蛋白性能。”

### 图表顺序

1. Figure 1：进化支持度与现有 PLM 性能鸿沟（含控制变量）。
2. Figure 2：层级同源覆盖、采样器行为和数据质量审计。
3. Figure 3：ProteinGym 分层主结果及 head-retention Pareto 图。
4. Figure 4：远程结构/功能泛化与 ESG benchmark。
5. Figure 5：消融、尺度、稳健性；若有效再加入 EvoTail-Struct。

### 投稿决策

- 若只有单一生物基准优势：ISMB、RECOMB、Bioinformatics。
- 若有统一目标、严格基准、两类以上任务收益及机制分析：NeurIPS/ICLR 主会可尝试。
- 不以“EvoTail 三个技巧都有效”为论文主线；以“进化信息预算 + worst-group generalization”为主线。

## 10. 最终交付清单

- 版本化数据清单和所有 split；
- support score、采样权重和人工质检记录；
- token-matched baselines 及可复现实验配置；
- ProteinGym 原始/分层结果、置信区间和官方评分日志；
- EvoTail-Bench data card 与泄漏审计；
- 训练代码、评测脚本、模型卡、限制与伦理说明；
- 论文、补充材料、可复现实验表格。
