# OpenSearch-VL 面试准备指南

## 项目概述

**一句话介绍**：OpenSearch-VL 是一个完全开源的多模态深度搜索代理训练方案，通过 Fatal-Aware GRPO 算法和 Wikipedia 数据策展管道，训练能够使用多种工具（搜索、OCR、图像增强）进行多轮推理的 Visual Investigation Agent。

**核心成果**：
- 在 7 个知识密集型基准测试中平均提升 10+ 分
- 32B 模型超越 Gemini-2.5-Pro 和 GPT-4o
- 完全开源：数据、代码、模型权重

---

## 第一部分：技术架构深度解析

### 1.1 整体训练流程

```
┌────────────────┐     ┌────────────────┐     ┌────────────────────┐
│ Qwen3-VL base  │ ─── │ Agentic SFT    │ ─── │ Async Agentic RL   │ ───▶ OpenSearch-VL
│ (HF weights)   │     │ (LLaMA-Factory)│     │ (rLLM + verl)      │
└────────────────┘     └────────────────┘     └────────────────────┘
                             │                         │
                             ▼                         ▼
                     SearchVL-SFT-36k          SearchVL-RL-8k
                     36,592条轨迹              8,000条RL样本
                     平均6.3轮工具调用         Fatal-Aware GRPO
```

**面试考察点**：
- 为什么需要 SFT 阶段？直接 RL 可以吗？
- SFT 和 RL 的数据来源有什么区别？
- 为什么 RL 数据要和 SFT 数据 disjoint？

### 1.2 工具环境设计

| 类别 | 工具 | 用途 |
|------|------|------|
| **检索** | `text_search`, `image_search`, `web_search`, `visit` | 获取外部文本/视觉证据 |
| **图像增强** | `sharpen`, `super_resolution`, `perspective_correct` | 修复模糊、低分辨率输入 |
| **注意力与解析** | `crop`, `layout_parsing` (OCR) | 定位感兴趣区域 |
| **计算** | `python_interpreter` | 数值/程序化计算 |

**面试考察点**：
- 为什么需要图像增强工具？直接用原始图像不行吗？
- 工具调用的格式是什么？如何解析？
- 如何处理工具执行失败的情况？

---

## 第二部分：核心算法 - Fatal-Aware GRPO

### 2.1 算法背景

**问题**：多轮工具使用中，代理经常遇到"致命"状态（如级联工具失败、无限循环），后续推理变得无意义。

**传统方法的局限**：
- **Hard Masking**（Vision-DeepResearch）：丢弃整个轨迹，浪费有效前缀
- **Vanilla GRPO**：对整个轨迹优化，引入失败后噪声

### 2.2 Fatal-Aware GRPO 核心机制

#### 2.2.1 致命步骤检测

```python
_CONSECUTIVE_ERROR_THRESHOLD = 3

def _find_fatal_step_index(steps, termination):
    consecutive_errors = 0
    for idx, step in enumerate(steps):
        if _is_step_error(step):
            consecutive_errors += 1
        else:
            consecutive_errors = 0  # 恢复时重置计数器
        
        if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
            fatal_start = idx - _CONSECUTIVE_ERROR_THRESHOLD + 1
            return fatal_start, "consecutive_errors"
    
    if termination == "answer":
        return None, ""
    return len(steps), termination or "no_final_answer"
```

**关键设计点**：
1. **连续错误计数器**：只有连续 K=3 次错误才触发致命标记
2. **重置机制**：单个错误后的成功恢复会重置计数器
3. **优雅降级**：异常终止但无错误级联的轨迹，保留所有步骤

**面试深挖问题**：
- 为什么选择 K=3 而不是 K=1 或 K=5？
- 为什么需要重置机制？没有重置会怎样？
- 如何定义"工具执行错误"？

#### 2.2.2 Token 掩码

**数学公式**：
```
M(y_{i,t}) = M_gen(y_{i,t}) × 1[s(t) < f_i]
```

其中：
- `M_gen`：生成掩码（排除观察 token）
- `f_i`：致命步骤索引
- `s(t)`：token 索引到步骤索引的映射

**面试考察点**：
- 为什么要排除观察 token？包括观察 token 会怎样？
- 掩码是如何在训练中实现的？

#### 2.2.3 单侧优势钳位

**数学公式**：
```
Â_i = {
  r̃_i,                    if f_i = L_i + 1 (非致命)
  max(r̃_i, 0),           if f_i ≤ L_i (致命)
}
```

**核心思想**：
- 非致命轨迹：正常使用标准化奖励
- 致命轨迹：只保留正向奖励，负向奖励钳位为 0

**面试深挖问题**：
- 为什么需要单侧钳位？直接用 hard masking 不行吗？
- 钳位会引入偏差吗？如何缓解？
- 如果致命轨迹的前缀质量很高，但整体奖励很低，会发生什么？

### 2.3 复合奖励设计

**数学公式**：
```
r(τ) = r_fmt(τ) × [α × r_acc(τ) + (1-α) × r_query(τ)]
```

其中 α=0.8

**三个奖励组件**：

1. **格式奖励 r_fmt ∈ [0,1]**：
   - 检查 `<think>...</think>` + `<tool_call>...</tool_call>` 或 `<response>...</response>`
   - 作为乘性门，格式错误的轨迹奖励趋近于 0

2. **准确性奖励 r_acc ∈ {0,1]**：
   - GPT-4o 判断最终答案是否正确
   - 致命轨迹强制 r_acc=0

3. **查询质量奖励 r_query ∈ [0,1]**：
   - GPT-5.4 评估搜索轨迹质量
   - 四个维度：相关性、逻辑进展、信噪比、跨模态互补性

**面试深挖问题**：
- 为什么 r_fmt 是乘性而不是加性？
- 为什么 α=0.8？如何选择这个值？
- r_query 对于致命轨迹如何处理？

---

## 第三部分：数据策展管道

### 3.1 Wikipedia 路径采样

**核心思想**：从 Wikipedia 超链接图中采样多跳路径，构建需要多步推理的 VQA。

**路径长度分布**：h ∈ {2, 3, 4}，概率分别为 (0.4, 0.4, 0.2)

**过滤规则**：
1. 跳过消歧页面和列表页面
2. 跳过循环
3. 跳过入度 > 10,000 的枢纽节点
4. 跳过非文章命名空间

**种子节点要求**：
1. 必须有信息框
2. 必须有至少一张 ≥ 512×512 的 Wikimedia Commons 图像
3. 入度必须在 [50, 10,000] 范围内

**面试深挖问题**：
- 为什么限制路径长度为 2-4 跳？
- 为什么要跳过枢纽节点？
- 如何确保采样的路径有多样性？

### 3.2 模糊实体重写

**核心思想**：用描述符替换实体名称，防止单跳检索捷径。

**三个不变量**：
1. **答案不变性**：a(q_f) = a(q_t)
2. **唯一性**：|R(q_f)| = 1
3. **非泄漏**：∪ aliases(v_j) ∩ q_f = ∅

**重写过程**：
1. 从最远的桥节点 v_{h-1} 开始，向 v_0 推进
2. 每次只重写一个实体
3. 使用 GPT-4o 生成描述符
4. 使用 LLM 唯一性评估器验证

**面试深挖问题**：
- 为什么要从最远的桥节点开始重写？
- 如何确保重写后的描述符是唯一的？
- 如果多个实体有相同的描述符怎么办？

### 3.3 源锚视觉接地

**核心创新**：图像来自源节点 v_0，而非答案节点 v_h。

**效果**：
- 防止单跳图像查找捷径
- 代理必须先识别视觉锚点，然后跟随中间文本关系

**示例**：
- 路径：Australia Zoo → Steve Irwin → Terri Irwin
- 问题："图中动物园管理者的妻子何时获得澳大利亚公民身份？"
- 图像：Australia Zoo 的照片（而非 Terri Irwin 的照片）

**面试深挖问题**：
- 为什么图像要来自源节点而不是答案节点？
- 如何选择"代表性"图像？
- CLIP 相似度阈值 0.28 是如何确定的？

### 3.4 两阶段过滤

**第一阶段**：过滤无需工具即可回答的样本
- 使用冻结的 Qwen3-VL-32B
- 丢弃仅依赖参数知识的样本

**第二阶段**：过滤单次搜索即可解决的样本
- 丢弃仅需一次 ImageSearch 调用的样本
- 确保保留的样本需要真正的多跳推理

**面试深挖问题**：
- 为什么需要两阶段过滤？
- 如何判断样本是否需要工具？
- 过滤比例是多少？

### 3.5 增强子集

**目的**：训练代理处理视觉不完美的能力

**增强类型**：
1. 模糊（Blur）→ Sharpen
2. 降采样（Downsampling）→ SuperResolution
3. 透视畸变（Perspective Distortion）→ PerspectiveCorrect

**比例**：过滤后池的 10%

**面试深挖问题**：
- 为什么只对 10% 的样本做增强？
- 如何确保增强后的样本仍然可以回答？

---

## 第四部分：工程实现细节

### 4.1 异步架构

**组件**：
1. **Rollout Engine**：异步生成轨迹
2. **Tool Executor**：异步执行工具调用
3. **Reward Function**：异步计算奖励
4. **Trainer**：异步更新策略

**优势**：
- 轨迹生成和策略更新解耦
- 工具调用不阻塞模型推理
- 可以并行处理多个任务

**面试深挖问题**：
- 如何保证异步训练的稳定性？
- 如果工具调用超时怎么办？
- 如何处理网络波动？

### 4.2 内存优化技术

**CPU Offload**：
- 参数、优化器状态、梯度都卸载到 CPU 内存
- GPU 只存储激活值和当前计算所需的参数

**Gradient Checkpointing**：
- 使用全重新计算（每 1 层）
- 用计算换显存

**DeepSpeed ZeRO-3**：
- 将模型参数、梯度、优化器状态分片到所有 GPU

**面试深挖问题**：
- CPU Offload 的性能开销是多少？
- Gradient Checkpointing 如何选择重计算的粒度？
- ZeRO-3 和 ZeRO-2 有什么区别？

### 4.3 分布式训练

**Megatron-LM 并行**：
- **TP=4**：将矩阵乘法分片到 4 个 GPU
- **PP=2**：将模型层分到 2 个阶段
- **CP=8**：将序列分片到 8 个 GPU

**SGLang Rollout**：
- TP=4
- GPU 显存利用率 0.85

**面试深挖问题**：
- TP、PP、CP 分别适合什么场景？
- 如何选择并行策略？
- SGLang 和 vLLM 有什么区别？

---

## 第五部分：面试常见问题

### 5.1 算法理解类

**Q1：Fatal-Aware GRPO 和 Vanilla GRPO 的核心区别是什么？**

**A1**：
1. **环境泛化**：Vanilla GRPO 是单轮生成，Fatal-Aware GRPO 支持多轮多模态环境
2. **掩码扩展**：Vanilla GRPO 无掩码，Fatal-Aware GRPO 有致命感知掩码
3. **优势计算**：Vanilla GRPO 使用组标准化奖励，Fatal-Aware GRPO 使用复合奖励 + 单侧钳位
4. **奖励设计**：Vanilla GRPO 是单一奖励，Fatal-Aware GRPO 是复合奖励

**Q2：为什么需要单侧优势钳位？**

**A2**：
- 直接使用标准化奖励会惩罚致命轨迹的有效前缀
- 单侧钳位确保有效前缀只被强化，不被惩罚
- 对于负向奖励的致命轨迹，钳位为 0，避免噪声梯度

**Q3：复合奖励中为什么 r_fmt 是乘性而不是加性？**

**A3**：
- 格式错误的轨迹应该被严重惩罚
- 乘性门可以将格式错误的轨迹奖励趋近于 0
- 如果是加性，格式错误但答案正确的轨迹仍会获得高奖励

### 5.2 数据工程类

**Q4：源锚视觉接地的核心创新是什么？**

**A4**：
- 传统方法：图像来自答案节点，可以通过单跳图像搜索找到答案
- 创新方法：图像来自源节点，代理必须先识别视觉锚点，再通过多步推理找到答案
- 效果：防止单跳检索捷径，确保需要真正的多跳推理

**Q5：模糊实体重写的三个不变量是什么？**

**A5**：
1. **答案不变性**：重写后答案不变
2. **唯一性**：重写后的描述符只能指向一个实体
3. **非泄漏**：重写后的问题不包含任何实体名称或别名

**Q6：两阶段过滤的作用是什么？**

**A6**：
- 第一阶段：过滤无需工具即可回答的样本（仅依赖参数知识）
- 第二阶段：过滤单次搜索即可解决的样本（不需要多跳推理）
- 确保保留的样本真正需要多跳推理和工具使用

### 5.3 工程实现类

**Q7：异步架构的优势是什么？**

**A7**：
- 轨迹生成和策略更新解耦，提高 GPU 利用率
- 工具调用不阻塞模型推理，可以并行处理多个任务
- 更好的容错性，单个任务失败不影响其他任务

**Q8：如何处理工具执行失败？**

**A8**：
- 连续错误计数器：单个错误后的成功恢复会重置计数器
- 致命步骤检测：连续 K=3 次错误触发致命标记
- Token 掩码：掩码致命步骤之后的所有 token
- 单侧钳位：只强化有效前缀，不惩罚

**Q9：如何优化内存使用？**

**A9**：
1. **CPU Offload**：将优化器状态和梯度卸载到 CPU
2. **Gradient Checkpointing**：用计算换显存
3. **DeepSpeed ZeRO-3**：将模型参数、梯度、优化器状态分片
4. **动态批量大小**：根据序列长度动态调整
5. **序列长度缩减**：从 32000 减少到 8192

### 5.4 系统设计类

**Q10：如何设计一个多轮 Agent 的训练系统？**

**A10**：
1. **数据策展**：构建高质量的多轮轨迹数据
2. **SFT 冷启动**：用监督微调初始化代理行为
3. **RL 探索**：用强化学习发现更好的策略
4. **奖励设计**：结合结果奖励和过程奖励
5. **错误处理**：处理级联失败和工具错误
6. **分布式训练**：支持大规模模型训练

**Q11：如何评估 Agent 的性能？**

**A11**：
- **准确性**：最终答案是否正确
- **效率**：工具调用次数和轮次
- **鲁棒性**：处理错误和异常的能力
- **泛化性**：在未见数据上的表现

---

## 第六部分：项目介绍模板

### 6.1 一分钟版本

> 我复现了 OpenSearch-VL 项目，这是一个多模态深度搜索代理训练方案。核心创新是 Fatal-Aware GRPO 算法，通过检测连续工具失败（K=3）并掩码失败后的 token，同时使用单侧优势钳位保留有效前缀。数据方面，通过 Wikipedia 路径采样、模糊实体重写和源锚视觉接地，构建真正需要多跳推理的训练数据。在 7 个基准测试中平均提升 10+ 分。

### 6.2 三分钟版本

> 我复现了 OpenSearch-VL 项目，这是一个完全开源的多模态深度搜索代理训练方案。
>
> **技术架构**：项目采用两阶段训练：首先用 SFT 冷启动，然后用 RL 探索更优策略。SFT 阶段使用 36,592 条专家轨迹，平均 6.3 轮工具调用；RL 阶段使用 8,000 条样本，采用 Fatal-Aware GRPO 算法。
>
> **核心创新**：
> 1. **Fatal-Aware GRPO**：解决级联工具失败问题。通过连续错误计数器检测致命步骤（K=3），掩码失败后的 token，并使用单侧优势钳位保留有效前缀。
> 2. **复合奖励设计**：r = r_fmt × (0.8 × r_acc + 0.2 × r_query)，平衡格式、准确性和查询质量。
> 3. **数据策展管道**：通过 Wikipedia 路径采样、模糊实体重写和源锚视觉接地，构建真正需要多跳推理的训练数据。
>
> **实验结果**：在 7 个知识密集型基准测试中平均提升 10+ 分，32B 模型超越 Gemini-2.5-Pro 和 GPT-4o。
>
> **我的工作**：分析了计算资源需求，设计了针对 8×3090/4090 的小规模复现方案，完成了推理验证和 LoRA 微调实验。

### 6.3 五分钟版本（技术深度）

> 我复现了 OpenSearch-VL 项目，这是一个完全开源的多模态深度搜索代理训练方案。
>
> **问题背景**：多模态搜索代理需要处理复杂的多跳推理和工具使用，但传统 RL 方法在处理级联工具失败时存在问题：要么丢弃整个轨迹（浪费有效前缀），要么对整个轨迹优化（引入噪声）。
>
> **核心创新**：
>
> 1. **Fatal-Aware GRPO 算法**：
>    - **致命步骤检测**：使用连续错误计数器，只有连续 K=3 次工具执行错误才触发致命标记
>    - **Token 掩码**：M(y_{i,t}) = M_gen(y_{i,t}) × 1[s(t) < f_i]
>    - **单侧优势钳位**：Â_i = max(r̃_i, 0) for fatal trajectories
>    - **复合奖励**：r = r_fmt × (0.8 × r_acc + 0.2 × r_query)
>
> 2. **数据策展管道**：
>    - **Wikipedia 路径采样**：从超链接图采样 2-4 跳路径
>    - **模糊实体重写**：用描述符替换实体名，三个不变量（答案不变性、唯一性、非泄漏）
>    - **源锚视觉接地**：图像来自源节点而非答案节点，防止单跳捷径
>    - **两阶段过滤**：过滤无需工具即可回答的样本
>
> 3. **工程实现**：
>    - **异步架构**：轨迹生成和策略更新解耦
>    - **内存优化**：CPU Offload + Gradient Checkpointing + DeepSpeed ZeRO-3
>    - **分布式训练**：Megatron-LM TP/PP/CP + SGLang 推理引擎
>
> **实验结果**：在 7 个基准测试中平均提升 10+ 分。消融研究表明，源锚接地贡献最大（-11.5），模糊实体重写次之（-10.3），Fatal-Aware GRPO 显著优于 Hard Masking（+4.1）。
>
> **我的工作**：
> 1. 深入分析了 Fatal-Aware GRPO 算法实现，理解了连续错误检测、单侧优势钳位、复合奖励设计
> 2. 设计了针对 8×3090/4090 的小规模复现方案，通过 LoRA 微调、CPU Offload 等技术，在 8B 模型上实现 5-10% 性能损失的可控复现
> 3. 编写了完整的复现文档，涵盖配置分析、资源评估、适配方案和实施步骤

---

## 第七部分：面试深挖问题集

### 7.1 算法深挖

**Q1：为什么选择 GRPO 而不是 PPO？**

**A1**：
- GRPO 不需要单独的 critic 模型，节省内存
- GRPO 使用组内比较计算优势，更适合稀疏奖励场景
- GRPO 的 KL 正则化更稳定

**Q2：如果 K=1 会发生什么？**

**A2**：
- 任何单个工具错误都会触发致命标记
- 大量轨迹会被标记为致命，训练信号稀疏
- 代理可能学会避免使用工具，降低探索能力

**Q3：为什么 r_query 只占 0.2？**

**A3**：
- 准确性是最终目标，应该占主导
- 查询质量是过程信号，用于提供密集反馈
- 如果 r_query 权重太高，代理可能优化搜索策略但忽略答案正确性

### 7.2 数据工程深挖

**Q4：如何确保模糊实体重写的唯一性？**

**A4**：
- 使用 LLM 唯一性评估器
- 给定重写后的问题，评估器判断是否只有一个实体符合描述
- 如果多个实体符合，重新生成描述符

**Q5：CLIP 相似度阈值 0.28 是如何确定的？**

**A5**：
- 通过实验确定，在准确性和召回率之间权衡
- 太低：图像与问题不相关
- 太高：过滤掉太多有效样本

**Q6：如果 Wikipedia 路径采样失败怎么办？**

**A6**：
- 最多重试 10 次
- 如果仍然失败，丢弃该种子
- 使用分层采样确保多样性

### 7.3 工程实现深挖

**Q7：异步训练如何保证一致性？**

**A7**：
- 使用版本化的策略模型
- Rollout 使用旧版本策略生成轨迹
- Trainer 使用新版本策略更新参数
- 通过 KL 正则化防止策略偏移太大

**Q8：CPU Offload 的性能开销是多少？**

**A8**：
- 数据传输开销：PCIe 带宽限制
- 训练速度降低：约 20-30%
- 但可以训练更大的模型或使用更大的批量

**Q9：如何处理工具调用超时？**

**A9**：
- 设置超时时间（如 30 秒）
- 超时后返回错误信息
- 连续超时触发致命标记
- 使用重试机制处理临时故障

### 7.4 系统设计深挖

**Q10：如何扩展到更大的模型（如 70B）？**

**A10**：
- 增加 TP/PP 并行度
- 使用更激进的 CPU Offload
- 考虑使用 MoE 架构
- 可能需要更多节点

**Q11：如何处理多模态输入的对齐问题？**

**A11**：
- 使用 M-RoPE（多分辨率位置编码）
- 图像 token 和文本 token 使用不同的位置编码
- 通过视觉编码器提取图像特征
- 使用投影层对齐视觉和文本特征

**Q12：如何评估 Agent 的泛化能力？**

**A12**：
- 在多个基准测试上评估
- 测试不同类型的工具使用
- 评估在未见数据上的表现
- 分析失败案例，找出改进方向

---

## 第八部分：关键代码片段

### 8.1 致命步骤检测

```python
_CONSECUTIVE_ERROR_THRESHOLD = 3

def _find_fatal_step_index(steps, termination):
    consecutive_errors = 0
    for idx, step in enumerate(steps):
        if _is_step_error(step):
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        
        if consecutive_errors >= _CONSECUTIVE_ERROR_THRESHOLD:
            fatal_start = idx - _CONSECUTIVE_ERROR_THRESHOLD + 1
            return fatal_start, "consecutive_errors"
    
    if termination == "answer":
        return None, ""
    return len(steps), termination or "no_final_answer"
```

### 8.2 工具错误检测

```python
def _has_tool_error_observation(observation):
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "[Python Interpreter Error]",
        "Python execution error:",
        "Tool execution error:",
        "Error executing",
        "Error: Image reference",
        "Error: OpenCV not available",
    )
    return any(marker in observation for marker in error_markers)
```

### 8.3 奖励计算

```python
# 奖励组成
r_accuracy = 1.0 if episode.is_correct else 0.0
r_format = sum(fmt_scores) / len(fmt_scores)
r_query = await _judge_query_utility(question, answer, prediction, steps)

total_reward = r_format * (0.8 * r_accuracy + 0.2 * r_query)

# 致命轨迹处理
if is_fatal and fatal_step_index == 0:
    trajectory.reward = 0.0
else:
    scored_steps = all_steps[:fatal_step_index] if is_fatal else all_steps
    # 只在有效前缀上计算奖励
```

### 8.4 格式验证

```python
def _is_valid_format(content):
    if not isinstance(content, str) or not content:
        return False
    pattern = r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>)\s*$"
    return re.match(pattern, content, re.DOTALL) is not None
```

---

## 第九部分：学习资源

### 9.1 论文

- **OpenSearch-VL**: https://arxiv.org/pdf/2605.05185
- **Search-R1**: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning
- **Vision-DeepResearch**: Multi-turn Deep Research with Visual Tools
- **GRPO**: Group Relative Policy Optimization (DeepSeek-Math)

### 9.2 代码

- **项目代码**: /home/chenyizhou/OpenSearch-VL
- **学习笔记**: /home/chenyizhou/OpenSearch-VL/doc/
- **官方文档**: https://github.com/shawn0728/OpenSearch-VL

### 9.3 技术栈

- **训练框架**: LLaMA-Factory, rLLM, verl
- **分布式训练**: Megatron-LM, DeepSpeed ZeRO-3
- **推理引擎**: SGLang
- **模型架构**: Qwen3-VL, M-RoPE
- **算法**: GRPO, Fatal-Aware GRPO, PPO

---

## 第十部分：面试准备清单

### 10.1 必须掌握的概念

- [ ] Fatal-Aware GRPO 算法的核心机制
- [ ] 复合奖励设计的三个组件
- [ ] 数据策展管道的三个阶段
- [ ] 源锚视觉接地的创新点
- [ ] 模糊实体重写的三个不变量
- [ ] 异步架构的优势
- [ ] 内存优化技术

### 10.2 必须准备的问题

- [ ] 为什么需要 Fatal-Aware GRPO？
- [ ] 单侧优势钳位的作用是什么？
- [ ] 源锚视觉接地如何防止单跳捷径？
- [ ] 如何处理工具执行失败？
- [ ] 如何优化内存使用？
- [ ] 如何设计多轮 Agent 的训练系统？

### 10.3 必须熟悉的代码

- [ ] 致命步骤检测逻辑
- [ ] 工具错误检测逻辑
- [ ] 奖励计算逻辑
- [ ] 格式验证逻辑
- [ ] 多轮 Agent 循环

---

## 总结

OpenSearch-VL 是一个技术深度很高的项目，涵盖了：
1. **算法创新**：Fatal-Aware GRPO、复合奖励设计
2. **数据工程**：Wikipedia 路径采样、模糊实体重写、源锚视觉接地
3. **工程实现**：异步架构、内存优化、分布式训练

面试时，重点展示：
1. **对算法的深入理解**：不只是知道是什么，还要知道为什么
2. **对数据工程的理解**：如何构建高质量的训练数据
3. **对工程实现的理解**：如何处理大规模训练的挑战
4. **对问题的思考能力**：如何分析问题、设计方案、验证效果
