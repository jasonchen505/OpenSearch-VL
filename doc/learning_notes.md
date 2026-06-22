# OpenSearch-VL 学习笔记

## 1. 项目概述

OpenSearch-VL 是一个完全开源的多模态深度搜索代理训练方案，基于强化学习训练前沿的多模态搜索代理。与标准 VLM 不同，该代理以闭环方式运行：检查图像、裁剪或增强感兴趣区域、发起网络和图像搜索、访问检索到的页面，然后基于收集到的证据撰写答案。

### 核心创新点

1. **数据策展管道**：基于 Wikipedia 超链接图构建，合成图像接地的多跳 VQA
2. **工具环境**：统一的视觉和检索工具环境（crop、layout_parsing、text_search、image_search 等）
3. **算法**：多轮 fatal-aware GRPO 算法，显式处理长 rollout 中的级联工具失败

### 关键结果

在 7 个知识密集型多模态基准测试中，OpenSearch-VL 平均得分提升超过 **10 分**，30B/32B 规模模型匹配强专有系统的准确率。

---

## 2. 技术架构

### 2.1 模型变体

| 变体 | 类型 | 参数量 | 基础模型 |
|------|------|--------|----------|
| OpenSearch-VL-8B | Dense | 8B | Qwen3-VL-8B-Instruct |
| OpenSearch-VL-30B-A3B | MoE | 30B (3B 激活) | Qwen3-VL-30B-A3B-Instruct |
| OpenSearch-VL-32B | Dense | 32B | Qwen3-VL-32B-Instruct |

### 2.2 训练流程

```
┌────────────────┐     ┌────────────────┐     ┌────────────────────┐
│ Qwen3-VL base  │ ─── │ Agentic SFT    │ ─── │ Async Agentic RL   │ ───▶ OpenSearch-VL
│ (HF weights)   │     │ (code/SFT)     │     │ (code/RL)          │
└────────────────┘     └────────────────┘     └────────────────────┘
                             │                         │
                             ▼                         ▼
                     SearchVL-SFT-36k          SearchVL-RL-8k
                     7-domain tool-use         Vision-DeepResearch-QA
                     cold-start trajectories   (RLOO / GRPO + fatal-aware)
```

### 2.3 工具环境

| 类别 | 工具 | 用途 |
|------|------|------|
| **检索** | `text_search`, `image_search`, `web_search`, `visit` | 获取外部文本/视觉证据并访问页面 |
| **图像增强** | `sharpen`, `super_resolution`, `perspective_correct` | 修复模糊、低分辨率或倾斜的输入 |
| **注意力与解析** | `crop`, `layout_parsing` (OCR) | 定位感兴趣区域并解码细粒度内容 |
| **计算** | `python_interpreter` | 对检索到的证据进行数值/程序化计算 |

---

## 3. 计算资源需求分析

### 3.1 官方资源配置

#### SFT 阶段
| 模型 | GPU 配置 | 训练时间 | 备注 |
|------|----------|----------|------|
| 8B | 256 张 H20 (32×8) | ~2 天 | 全参数微调，ZeRO-3 |
| 30B-A3B | 256 张 H20 (32×8) | ~4 天 | 全参数微调，ZeRO-3 |

#### RL 阶段
| 模型 | GPU 配置 | 训练时间 | 备注 |
|------|----------|----------|------|
| 8B | 64 张 H20 (8×8) | ~10 天 | 200 步，SGLang + Megatron |
| 30B-A3B | 64 张 H20 (8×8) | — | 5 个 epoch |

### 3.2 关键训练参数

#### SFT 超参数
- **Learning rate**: 2.0 × 10⁻⁵
- **Epochs**: 8
- **LR scheduler**: cosine with 0.1 warmup ratio
- **Effective batch size**: 256 (1 per device × 1 grad accum × 256 GPUs)
- **Max sequence length**: 32,000 tokens
- **Gradient checkpointing**: True
- **Mixed precision**: bfloat16
- **Optimizer**: AdamW (DeepSpeed ZeRO-3)

#### RL 超参数 (8B)
- **Samples per prompt (G)**: 8
- **Actor TP / PP / CP**: 4 / 2 / 8
- **PPO mini-batch size**: 64
- **Max response length**: 70,000 tokens
- **Actor learning rate**: 1×10⁻⁶
- **PPO clip ratio (high)**: 0.28
- **KL coefficient**: 1×10⁻³
- **Entropy coefficient**: 0.0
- **Temperature (train/val)**: 0.7 / 0.7
- **Total epochs**: 100
- **Gradient checkpointing**: Full recompute (1 layer)
- **Param/optim/grad offload**: CPU
- **Rollout engine**: SGLang (async)
- **Rollout TP**: 4

### 3.3 显存需求估算

#### H20 GPU 规格
- **显存**: 96GB HBM3
- **计算能力**: 96 TFLOPS (FP16)
- **显存带宽**: 4TB/s

#### 8B 模型显存需求（估算）
- **模型参数**: 8B × 2 bytes (bf16) = 16GB
- **优化器状态**: 8B × 8 bytes (AdamW) = 64GB
- **梯度**: 8B × 2 bytes = 16GB
- **激活值**: 取决于序列长度和批量大小
- **总计**: 约 100-120GB（不含激活值）

#### 30B-A3B 模型显存需求（估算）
- **模型参数**: 30B × 2 bytes (bf16) = 60GB
- **优化器状态**: 30B × 8 bytes = 240GB
- **梯度**: 30B × 2 bytes = 60GB
- **总计**: 约 360GB（不含激活值）

---

## 4. 用户资源评估

### 4.1 用户 GPU 资源

| GPU 型号 | 数量 | 单卡显存 | 总显存 | 计算能力 |
|----------|------|----------|--------|----------|
| RTX 3090 | 8 | 24GB | 192GB | 35.6 TFLOPS (FP16) |
| RTX 4090 | 8 | 24GB | 192GB | 82.6 TFLOPS (FP16) |
| **总计** | **16** | — | **384GB** | — |

### 4.2 资源差距分析

| 对比项 | 用户资源 | 官方资源 | 差距 |
|--------|----------|----------|------|
| **SFT 8B** | 384GB | 24,576GB (256×96GB) | **64 倍** |
| **RL 8B** | 384GB | 6,144GB (64×96GB) | **16 倍** |
| **单卡显存** | 24GB | 96GB | **4 倍** |
| **GPU 数量** | 16 | 256 (SFT) / 64 (RL) | **16 倍 / 4 倍** |

### 4.3 可行性评估

#### 直接复现：❌ 不可行
- 显存不足：24GB 无法容纳 8B 模型的优化器状态
- GPU 数量不足：16 张卡无法达到 256 张卡的并行度
- 训练时间不可接受：即使能运行，训练时间将超过数月

#### 缩小规模复现：⚠️ 有限可行
- **只训练 8B 模型**：30B/32B 模型完全不可行
- **使用 CPU Offload**：将优化器状态和梯度卸载到 CPU 内存
- **减少序列长度**：从 32000 减少到 8192 或更小
- **减少批量大小**：使用 gradient accumulation
- **使用 LoRA/QLoRA**：减少可训练参数

#### 推理验证：✅ 可行
- **加载预训练模型进行推理**：使用官方发布的 checkpoint
- **评估模型性能**：在基准测试上验证结果
- **学习代码实现**：理解算法和工具实现

---

## 5. 可行的复现方案

### 方案 1：推理验证（推荐）

**目标**：验证官方模型性能，学习代码实现

**步骤**：
1. 下载官方 checkpoint（OpenSearch-VL-8B）
2. 配置推理环境
3. 在基准测试上运行推理
4. 分析结果和工具使用轨迹

**资源需求**：
- 1-2 张 4090 即可
- 约 50GB 显存（模型 + 推理）

**优点**：
- 资源需求低
- 可以验证论文结果
- 学习完整的推理流程

### 方案 2：LoRA 微调（有限复现）

**目标**：使用 LoRA 微调 8B 模型

**步骤**：
1. 准备 SFT 数据集（SearchVL-SFT-36k）
2. 配置 LlamaFactory + LoRA
3. 在 8 张 4090 上微调
4. 评估微调后的模型

**资源需求**：
- 8 张 4090
- 约 180GB 总显存
- 训练时间：约 1-2 周

**配置调整**：
```yaml
# 修改 SFT 配置
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 8192  # 减少序列长度
num_train_epochs: 3
```

**优点**：
- 资源需求显著降低
- 可以学习 SFT 流程
- 保持大部分性能

**缺点**：
- 无法完全复现论文结果
- 无法进行 RL 训练

### 方案 3：单节点 RL 实验（概念验证）

**目标**：在单节点上运行 RL 训练的简化版本

**步骤**：
1. 使用 LoRA 微调后的 8B 模型
2. 配置 rLLM + SGLang
3. 在 8 张 4090 上运行 RL
4. 分析训练动态

**资源需求**：
- 8 张 4090
- 约 180GB 总显存
- 训练时间：约 1 周

**配置调整**：
```bash
# 修改 RL 配置
NNODES=1
trainer.n_gpus_per_node=8
train_tp=2
train_pp=1
train_cp=4
gen_tp=2
train_prompt_bsz=32
n_resp_per_prompt=4
max_response_length=10000
```

**优点**：
- 可以学习 RL 训练流程
- 验证 fatal-aware GRPO 算法
- 理解多轮工具使用

**缺点**：
- 性能会显著低于官方结果
- 训练可能不稳定

### 方案 4：数据策展管道学习

**目标**：学习数据策展管道，理解数据生成过程

**步骤**：
1. 研究 Wikipedia 路径采样算法
2. 理解模糊实体重写
3. 学习源锚视觉接地
4. 分析多轮轨迹合成

**资源需求**：
- CPU 即可
- 无需 GPU

**优点**：
- 理解数据生成的核心创新
- 学习高质量 VQA 构建方法
- 可以应用到其他项目

---

## 6. 关键学习点

### 6.1 数据策展创新

1. **Wikipedia 路径采样**：从 Wikipedia 超链接图中采样多跳实体路径
2. **模糊实体重写**：使用 GPT-4o 将中间实体重写为模糊描述
3. **源锚视觉接地**：图像来自源节点而非答案节点，防止单跳捷径
4. **两阶段过滤**：先过滤无需工具即可回答的样本，再过滤单次搜索即可解决的样本

### 6.2 算法创新

1. **Fatal-aware token masking**：检测致命级联（K=3 次连续错误），掩码失败后的 token
2. **单侧优势钳位**：防止过早抑制可行的前缀，对致命轨迹的负优势置零
3. **复合奖励设计**：r_fmt（格式门）× [α·r_acc + (1-α)·r_query]
4. **多轮 GRPO**：在多轮工具使用场景中应用 GRPO

### 6.3 系统设计

1. **统一工具环境**：SFT、RL 和推理共享相同的工具接口
2. **异步 RL 训练**：使用 SGLang 进行异步 rollout
3. **Megatron-LM 并行**：TP + PP + CP 的混合并行策略
4. **CPU Offload**：将优化器状态和梯度卸载到 CPU

### 6.4 工程实践

1. **DeepSpeed ZeRO-3**：SFT 阶段的分布式训练
2. **Ray 集群管理**：多节点训练的编排
3. **Hydra 配置管理**：RL 训练的配置管理
4. **W&B 日志记录**：训练过程的监控

---

## 7. 总结

### 关键收获

1. **OpenSearch-VL 是一个完整的多模态搜索代理训练方案**，包含数据、工具、算法三个核心组件
2. **计算资源需求极高**，官方使用 256 张 H20 GPU 进行 SFT，64 张 H20 GPU 进行 RL
3. **用户资源（16 张 24GB GPU）严重不足**，无法直接复现论文结果
4. **可行的复现方案**包括：推理验证、LoRA 微调、单节点 RL 实验、数据策展学习

### 建议

1. **优先进行推理验证**，使用官方 checkpoint 验证性能
2. **学习数据策展管道**，理解高质量 VQA 的构建方法
3. **尝试 LoRA 微调**，在有限资源下学习 SFT 流程
4. **研究 RL 算法实现**，理解 fatal-aware GRPO 的细节

### 后续学习计划

1. 深入阅读论文原文，理解技术细节
2. 分析代码实现，学习工程实践
3. 尝试推理验证，验证模型性能
4. 应用到自己的项目中，借鉴创新点

---

## 参考资料

- **论文**: [OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents](https://arxiv.org/pdf/2605.05185)
- **代码**: [GitHub - OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL)
- **模型**: [HuggingFace - OpenSearch-VL](https://huggingface.co/OpenSearch-VL)
- **数据集**: [HuggingFace - OpenSearch-VL](https://huggingface.co/OpenSearch-VL)
