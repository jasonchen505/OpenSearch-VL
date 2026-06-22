# OpenSearch-VL 深入技术学习笔记（第二轮）

## 1. Fatal-Aware GRPO 算法深度解析

### 1.1 算法核心思想

Fatal-Aware GRPO 是 OpenSearch-VL 的核心创新之一，解决了多轮工具使用中的一个关键问题：**如何处理级联工具失败的轨迹**。

**传统方法的问题**：
- **Hard Masking**（Vision-DeepResearch）：直接丢弃整个致命轨迹，浪费了有效的前缀部分
- **Vanilla GRPO**：对整个轨迹进行优化，会将失败后的噪声引入训练

**Fatal-Aware GRPO 的解决方案**：
1. 检测致命步骤（K=3 次连续工具执行错误）
2. 掩码致命步骤之后的所有 token
3. 使用单侧优势钳位，只强化有效的前缀部分

### 1.2 致命步骤检测逻辑

**实现位置**：`/home/chenyizhou/OpenSearch-VL/RL/rllm/vision_deepresearch_async_workflow/deepresearch_workflow.py`

**核心代码**：
```python
_CONSECUTIVE_ERROR_THRESHOLD = 3

def _find_fatal_step_index(
    steps: list[Step], termination: str
) -> tuple[int | None, str]:
    """检测第一个致命步骤，使用连续错误计数器。
    
    返回 (fatal_step_index, reason)，其中 fatal_step_index 是要掩码的
    **第一个步骤**的 0-based 索引（该步骤及后续所有步骤将被掩码）。
    
    对于正常轨迹（以 answer 终止且无错误级联），返回 (None, "")。
    
    单个错误后的恢复会重置计数器——只有级联失败（>= 阈值的连续错误）
    才会触发掩码。
    """
    if not steps:
        return 0, "no_steps"

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

    # 异常终止但无错误级联——所有现有步骤可用但轨迹不完整
    # 在最后一步之后标记为致命，保留所有步骤，但仍为优势钳位标记轨迹
    return len(steps), termination or "no_final_answer"
```

**关键设计点**：
1. **连续错误计数器**：只有连续 K=3 次错误才触发致命标记
2. **重置机制**：单个错误后的成功恢复会重置计数器
3. **优雅降级**：对于异常终止但无错误级联的轨迹，保留所有步骤

### 1.3 工具执行错误检测

**错误标记器**：
```python
def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
        "[Python Interpreter Error]",
        "Python execution error:",
        "PythonInterpreter tool not available",
        "PythonInterpreter tool is not callable",
        "Tool execution error:",
        "Error executing",
        "Error: Image reference",
        "Error: OpenCV not available",
    )
    return any(marker in observation for marker in error_markers)
```

**步骤错误检测**：
```python
def _is_step_error(step: Step) -> bool:
    if step.info.get("step_error"):
        return True
    return _has_tool_error_observation(step.observation)
```

### 1.4 奖励计算与致命轨迹处理

**奖励组成**：
```python
r_total = r_format * (0.8 * r_accuracy + 0.2 * r_query_utility)
```

**致命轨迹的特殊处理**：
```python
if is_fatal and fatal_step_index == 0:
    # 无可用部分——退化为硬掩码
    episode.termination_reason = TerminationReason.UNKNOWN
    episode.info["fatal_step_index"] = 0
    episode.info["is_fatal"] = True
    episode.info["fatal_reason"] = fatal_reason
    for trajectory in episode.trajectories:
        trajectory.reward = 0.0
else:
    # 计算完整奖励用于组优势统计
    # 对于致命轨迹，r_format 和 r_query 仅在可学习前缀（致命前步骤）上评分
    r_accuracy = 1.0 if episode.is_correct else 0.0

    scored_steps = (
        all_steps[:fatal_step_index]
        if is_fatal and fatal_step_index < len(all_steps)
        else all_steps
    )
    if scored_steps:
        fmt_scores = [_format_reward_for_step(s) for s in scored_steps]
        r_format = sum(fmt_scores) / len(fmt_scores)
    else:
        r_format = 0.0

    query_steps = (
        all_steps[:fatal_step_index]
        if is_fatal and fatal_step_index < len(all_steps)
        else all_steps
    )
    r_query = await _judge_query_utility(
        question=question,
        ground_truth=answer,
        prediction=prediction,
        steps=query_steps,
    )

    total_reward = r_format * (
        QUERY_WEIGHT * r_query + ACCURACY_WEIGHT * r_accuracy
    )
```

### 1.5 单侧优势钳位（训练器实现）

**数学公式**：
```
Â_i = {
  r̃_i,                    if f_i = L_i + 1 (非致命)
  max(r̃_i, 0),           if f_i ≤ L_i (致命)
}
```

**实现位置**：训练器中（verl/rLLM 框架）

**关键逻辑**：
1. 所有轨迹（包括致命轨迹）都参与组统计计算
2. 致命轨迹的标准化奖励 r̃_i 如果为负，则钳位为 0
3. 致命轨迹的标准化奖励 r̃_i 如果为正，则保留原值
4. 这确保了有效前缀只被强化，不被惩罚

---

## 2. 多轮 Agent 循环实现

### 2.1 推理循环（pipeline.py）

**核心流程**：
```python
# 伪代码
for turn in range(max_turns):
    # 1. 模型生成响应
    response = model.generate(messages)
    
    # 2. 解析响应
    action = parse_action(response)
    
    # 3. 执行动作
    if action.type == "tool_call":
        observation = execute_tool(action.tool_call)
        messages.append({"role": "observation", "content": observation})
    elif action.type == "final_answer":
        return action.answer
    
    # 4. 更新历史
    messages.append({"role": "assistant", "content": response})
```

### 2.2 工具调用解析

**解析逻辑**：
```python
def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<response>" in response and "</response>" in response:
        answer = response.split("<response>")[1].split("</response>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})
```

### 2.3 格式验证

**验证逻辑**：
```python
def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<response>.*?</response>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None
```

**格式要求**：
1. 每个步骤必须以 `<think>` 开头
2. 之后必须是 `<tool_call>` 或 `<response>` 或 `<answer>`
3. 格式错误的步骤会被标记为 `step_error`

---

## 3. 数据策展管道深入分析

### 3.1 Wikipedia 路径采样

**路径长度分布**：h ∈ {2, 3, 4}，概率分别为 (0.4, 0.4, 0.2)

**过滤规则**：
1. 跳过消歧页面和列表页面
2. 跳过循环
3. 跳过入度 > 10,000 的枢纽节点
4. 跳过非文章命名空间（Template, Category, File 等）

**种子节点要求**：
1. 必须有信息框
2. 必须有至少一张 ≥ 512×512 的 Wikimedia Commons 图像
3. 入度必须在 [50, 10,000] 范围内

### 3.2 模糊实体重写

**三个不变量**：
1. **答案不变性**：a(q_f) = a(q_t)
2. **唯一性**：|R(q_f)| = 1
3. **非泄漏**：∪ aliases(v_j) ∩ q_f = ∅

**重写过程**：
1. 从最远的桥节点 v_{h-1} 开始，向 v_0 推进
2. 每次只重写一个实体
3. 使用 GPT-4o 生成描述符
4. 使用 LLM 唯一性评估器验证

### 3.3 源锚视觉接地

**关键创新**：图像来自源节点 v_0，而非答案节点 v_h

**效果**：
- 防止单跳图像查找捷径
- 代理必须先识别视觉锚点，然后跟随中间文本关系

**示例**：
- 路径：Australia Zoo → Steve Irwin → Terri Irwin
- 问题："图中动物园管理者的妻子何时获得澳大利亚公民身份？"
- 图像：Australia Zoo 的照片（而非 Terri Irwin 的照片）

### 3.4 两阶段过滤

**第一阶段**：过滤无需工具即可回答的样本
- 使用冻结的 Qwen3-VL-32B
- 丢弃仅依赖参数知识的样本

**第二阶段**：过滤单次搜索即可解决的样本
- 丢弃仅需一次 ImageSearch 调用的样本
- 确保保留的样本需要真正的多跳推理

### 3.5 增强子集

**目的**：训练代理处理视觉不完美的能力

**增强类型**：
1. 模糊（Blur）→ Sharpen
2. 降采样（Downsampling）→ SuperResolution
3. 透视畸变（Perspective Distortion）→ PerspectiveCorrect

**比例**：过滤后池的 10%

---

## 4. 系统提示词设计

### 4.1 代理系统提示词

**核心哲学**："Verify, Don't Guess"

**关键规则**：
1. **工具优先思维**：小文本 → crop；模糊 → sharpen；倾斜 → perspective_correct
2. **链接工具**：非平凡查询通常需要管道
3. **外部验证**：当答案依赖于非纯像素可见的事实时，必须调用 text_search

**思考协议**：
1. 分析请求
2. 评估图像质量（可读性、几何、目标大小）
3. 识别信息差距
4. 制定计划（承诺单一下一步行动）

### 4.2 工具使用规范

**输出规则**：
1. 每轮单一行动；等待结果后再进行下一步
2. 先思考：永远不要在没有 preceding <think> 的情况下发出 <tool_call>
3. 图像引用：初始图像为 img_1；每个工具输出产生 img_2, img_3, ...
4. 最终答案：一旦证据充足，发出 <response>...</response>

**工作流配方**：
- 不可读文档：perspective_correct → sharpen → layout_parsing
- 密集图表：crop（感兴趣区域）→ layout_parsing
- 实体识别：image_search → text_search（必须跟进）

---

## 5. 查询质量奖励（r_query）

### 5.1 评估维度

1. **图像搜索效用**：图像搜索是否检索到真正支持回答问题的视觉证据？
2. **文本搜索效用**：文本搜索是否找到相关信息？
3. **查询进展**：查询是否显示逻辑进展——精炼、缩小或覆盖不同方面？
4. **互补性**：图像和文本搜索是否相互补充？
5. **证据与噪声比**：检索结果中有多少实际包含有用证据？

### 5.2 评分标准

- **0.0**：未检索到有用信息；所有搜索无关或失败
- **0.3**：主要是噪声，偶尔有边际相关结果
- **0.5**：混合——找到一些有用证据但有显著噪声或低效
- **0.7**：良好搜索策略；大多数结果相关
- **1.0**：优秀——有针对性、高效的查询，检索到高度相关的证据

### 5.3 致命轨迹的特殊处理

对于致命轨迹，评估器仅评估有效前缀（致命前步骤），确保早期阶段的推理得到适当评价。

---

## 6. 与 Vanilla GRPO 的关键差异

### 6.1 环境泛化

**Vanilla GRPO**：单轮生成
**Search-R1 GRPO**：多轮，但仅文本检索器 R
**OpenSearch-VL GRPO**：多轮，多模态环境 E（视觉 + 检索工具）

### 6.2 掩码扩展

**Vanilla GRPO**：无掩码
**Search-R1 GRPO**：生成掩码 M_gen（排除观察 token）
**OpenSearch-VL GRPO**：致命感知掩码 M = M_gen × 1[s(t) < f_i]

### 6.3 优势计算

**Vanilla GRPO**：组标准化奖励
**Search-R1 GRPO**：组标准化奖励
**OpenSearch-VL GRPO**：复合奖励 + 单侧钳位

### 6.4 奖励设计

**Vanilla GRPO**：单一奖励
**Search-R1 GRPO**：单一奖励（答案正确性）
**OpenSearch-VL GRPO**：复合奖励 r_fmt × (0.8 × r_acc + 0.2 × r_query)

---

## 7. 实验结果分析

### 7.1 消融研究结果

**数据管道消融**（SimpleVQA + InfoSeek + FVQA 平均分）：
| 变体 | 平均分 | 变化 |
|------|--------|------|
| 完整管道 | **64.6** | — |
| 无源锚接地 | 53.1 | -11.5 |
| 无模糊实体重写 | 54.3 | -10.3 |
| 无分阶段过滤 | 56.4 | -8.2 |
| 无增强子集 | 63.3 | -1.3 |

**训练配方消融**：
| 方法 | 平均分 | vs Vanilla GRPO |
|------|--------|-----------------|
| 基础 Qwen3-VL-8B | 53.7 | — |
| + 仅 SFT | 64.6 | — |
| + Vanilla GRPO | 67.6 | 基线 |
| + Hard Masking | 67.7 | +0.1 |
| + Fatal Masking only | 69.1 | +1.5 |
| **+ Fatal Masking + One-sided Clamp** | **71.8** | **+4.2** |

### 7.2 关键发现

1. **源锚接地最重要**：移除导致 -11.5 分下降
2. **模糊实体重写次之**：移除导致 -10.3 分下降
3. **Fatal-aware GRPO 显著优于 Hard Masking**：+4.1 分
4. **单侧钳位提供额外提升**：+2.7 分 over Fatal Masking only

---

## 8. 工程实现细节

### 8.1 异步架构

**组件**：
1. **Rollout Engine**：异步生成轨迹
2. **Tool Executor**：异步执行工具调用
3. **Reward Function**：异步计算奖励
4. **Trainer**：异步更新策略

**优势**：
- 轨迹生成和策略更新解耦
- 工具调用不阻塞模型推理
- 可以并行处理多个任务

### 8.2 内存优化

**CPU Offload**：
- 参数、优化器状态、梯度都卸载到 CPU 内存
- GPU 只存储激活值和当前计算所需的参数

**Gradient Checkpointing**：
- 使用全重新计算（每 1 层）
- 用计算换显存

### 8.3 分布式训练

**Megatron-LM 并行**：
- TP=4：将矩阵乘法分片到 4 个 GPU
- PP=2：将模型层分到 2 个阶段
- CP=8：将序列分片到 8 个 GPU

**SGLang Rollout**：
- TP=4
- GPU 显存利用率 0.85

---

## 9. 总结与学习要点

### 9.1 核心创新

1. **Fatal-Aware Token Masking**：检测致命级联，掩码失败后的 token
2. **单侧优势钳位**：防止过早抑制可行的前缀
3. **复合奖励设计**：平衡格式、准确性和查询质量
4. **源锚视觉接地**：防止单跳捷径
5. **模糊实体重写**：确保真正需要多跳推理

### 9.2 工程实践

1. **异步架构**：轨迹生成和策略更新解耦
2. **内存优化**：CPU Offload + Gradient Checkpointing
3. **分布式训练**：Megatron-LM TP/PP/CP
4. **错误处理**：优雅降级和恢复机制

### 9.3 关键代码位置

| 组件 | 文件路径 |
|------|----------|
| Fatal 检测 | `RL/rllm/vision_deepresearch_async_workflow/deepresearch_workflow.py` |
| 奖励计算 | 同上 |
| 代理循环 | `opensearch_vl/opensearch_infer/pipeline.py` |
| 工具定义 | `opensearch_vl/opensearch_infer/tools.py` |
| 训练配置 | `SFT/examples/agentic_full/*.yaml` |
| RL 启动脚本 | `RL/rllm/vision_deepresearch_async_workflow/run/*.sh` |

---

## 参考资料

- **论文**：OpenSearch-VL: An Open Recipe for Frontier Multimodal Search Agents
- **代码**：/home/chenyizhou/OpenSearch-VL
- **论文 LaTeX**：/home/chenyizhou/OpenSearch-VL/OpenSearch-VL-paper-tex
