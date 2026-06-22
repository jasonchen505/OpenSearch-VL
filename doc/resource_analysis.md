# OpenSearch-VL 资源需求详细分析

## 1. 官方硬件配置

### 1.1 SFT 阶段硬件配置

| 模型 | GPU 配置 | 总 GPU 数 | 总显存 | 训练时间 |
|------|----------|-----------|--------|----------|
| 8B | 32 节点 × 8 卡 H20 | 256 | 24,576GB | ~2 天 |
| 30B-A3B | 32 节点 × 8 卡 H20 | 256 | 24,576GB | ~4 天 |
| 32B | 32 节点 × 8 卡 H20 | 256 | 24,576GB | ~4 天 |

### 1.2 RL 阶段硬件配置

| 模型 | GPU 配置 | 总 GPU 数 | 总显存 | 训练时间 |
|------|----------|-----------|--------|----------|
| 8B | 8 节点 × 8 卡 H100/H800 | 64 | 6,144GB | ~10 天 |
| 30B-A3B | 8 节点 × 8 卡 H100/H800 | 64 | 6,144GB | — |
| 32B | 16 节点 × 8 卡 H100/H800 | 128 | 12,288GB | — |

### 1.3 推理阶段硬件配置

| 模型 | 最低配置 | 推荐配置 | 备注 |
|------|----------|----------|------|
| 8B | 1 张 H100/A100-80GB | 1-4 张 GPU | 单卡可运行 |
| 30B-A3B | 4 张 GPU | 4 张 GPU | MoE 模型需要更多显存 |
| 32B | 2 张 GPU | 4-8 张 GPU | 密集模型 |

---

## 2. 显存需求详细分析

### 2.1 模型参数显存需求

#### 8B 模型
- **参数量**: 8B (8,000,000,000)
- **精度**: bfloat16 (2 bytes/param)
- **模型参数显存**: 8B × 2 bytes = **16GB**
- **优化器状态 (AdamW)**: 8B × 8 bytes = **64GB**
- **梯度**: 8B × 2 bytes = **16GB**
- **总计（不含激活值）**: **96GB**

#### 30B-A3B 模型
- **总参数量**: 30B
- **激活参数量**: 3B
- **精度**: bfloat16
- **模型参数显存**: 30B × 2 bytes = **60GB**
- **优化器状态**: 30B × 8 bytes = **240GB**
- **梯度**: 30B × 2 bytes = **60GB**
- **总计（不含激活值）**: **360GB**

#### 32B 模型
- **参数量**: 32B
- **精度**: bfloat16
- **模型参数显存**: 32B × 2 bytes = **64GB**
- **优化器状态**: 32B × 8 bytes = **256GB**
- **梯度**: 32B × 2 bytes = **64GB**
- **总计（不含激活值）**: **384GB**

### 2.2 激活值显存需求

激活值显存取决于：
- **序列长度**: 32,000 tokens (SFT), 70,000 tokens (RL response)
- **批量大小**: 1 (per device)
- **隐藏层大小**: 模型相关
- **层数**: 模型相关

**估算公式**:
```
激活值显存 ≈ 序列长度 × 批量大小 × 隐藏大小 × 层数 × 每层激活因子
```

**典型值**:
- 8B 模型: 约 20-40GB (取决于序列长度)
- 30B-A3B 模型: 约 60-120GB
- 32B 模型: 约 80-160GB

### 2.3 总显存需求估算

| 模型 | 模型参数 | 优化器状态 | 梯度 | 激活值 | 总计 |
|------|----------|------------|------|--------|------|
| 8B | 16GB | 64GB | 16GB | 20-40GB | **116-136GB** |
| 30B-A3B | 60GB | 240GB | 60GB | 60-120GB | **420-480GB** |
| 32B | 64GB | 256GB | 64GB | 80-160GB | **464-544GB** |

---

## 3. 并行策略分析

### 3.1 SFT 并行策略

**DeepSpeed ZeRO-3**:
- 将模型参数、梯度、优化器状态分片到所有 GPU
- 每个 GPU 只存储 1/N 的状态（N = GPU 数量）
- 通信开销较大，但显存效率高

**256 GPU 配置**:
- 每个 GPU 存储: 96GB / 256 = **0.375GB**（不含激活值）
- 加上激活值: 约 **0.5-0.6GB** per GPU
- 显存利用率: 约 **2-3%** of 24GB (3090/4090)

### 3.2 RL 并行策略

**Megatron-LM 并行**:
- **TP (Tensor Parallelism)**: 4 - 将矩阵乘法分片到 4 个 GPU
- **PP (Pipeline Parallelism)**: 2 - 将模型层分到 2 个阶段
- **CP (Context Parallelism)**: 8 - 将序列分片到 8 个 GPU

**64 GPU 配置 (8B)**:
- TP × PP × CP = 4 × 2 × 8 = 64
- 每个 GPU 存储: 96GB / 64 = **1.5GB**（不含激活值）
- 加上激活值: 约 **2-3GB** per GPU

**CPU Offload**:
- 参数、优化器状态、梯度都卸载到 CPU 内存
- GPU 只存储激活值和当前计算所需的参数
- 显存需求大幅降低，但训练速度变慢

---

## 4. 用户资源评估

### 4.1 用户 GPU 规格

| GPU 型号 | 数量 | 单卡显存 | 总显存 | FP16 算力 | 显存带宽 |
|----------|------|----------|--------|-----------|----------|
| RTX 3090 | 8 | 24GB | 192GB | 35.6 TFLOPS | 936 GB/s |
| RTX 4090 | 8 | 24GB | 192GB | 82.6 TFLOPS | 1,008 GB/s |
| **总计** | **16** | — | **384GB** | — | — |

### 4.2 与官方配置对比

| 对比项 | 用户资源 | 官方 SFT 8B | 官方 RL 8B | 差距 |
|--------|----------|-------------|------------|------|
| **GPU 数量** | 16 | 256 | 64 | **16x / 4x** |
| **单卡显存** | 24GB | 96GB (H20) | 96GB (H100) | **4x** |
| **总显存** | 384GB | 24,576GB | 6,144GB | **64x / 16x** |
| **FP16 算力** | 1,081.6 TFLOPS | 24,576 TFLOPS | 6,144 TFLOPS | **23x / 6x** |
| **显存带宽** | 15,552 GB/s | 1,024,000 GB/s | 256,000 GB/s | **66x / 16x** |

### 4.3 可行性评估

#### 直接复现官方配置：❌ 完全不可行

1. **显存不足**: 24GB 无法容纳 8B 模型的优化器状态 (64GB)
2. **GPU 数量不足**: 16 张卡无法达到 256 张卡的并行度
3. **训练时间不可接受**: 即使能运行，训练时间将超过数月
4. **通信瓶颈**: 消费级 GPU 缺乏 NVLink/InfiniBand，多卡通信效率低

#### 缩小规模复现：⚠️ 有限可行

**方案 A: LoRA 微调 8B 模型**
- 使用 LoRA 减少可训练参数
- 使用 CPU Offload 卸载优化器状态
- 减少序列长度 (32000 → 8192)
- 使用 gradient accumulation
- **预计显存**: 约 18-20GB per GPU
- **预计训练时间**: 1-2 周
- **性能损失**: 约 5-10%

**方案 B: QLoRA 微调 8B 模型**
- 使用 4-bit 量化 + LoRA
- 进一步减少显存需求
- **预计显存**: 约 12-15GB per GPU
- **预计训练时间**: 1-2 周
- **性能损失**: 约 10-15%

**方案 C: 单节点 RL 实验**
- 使用 LoRA 微调后的 8B 模型
- 单节点 8 卡配置
- 减少 batch size 和序列长度
- **预计显存**: 约 20-22GB per GPU
- **预计训练时间**: 1 周
- **性能损失**: 显著

#### 推理验证：✅ 完全可行

**方案 D: 推理验证**
- 使用官方发布的 checkpoint
- 单卡或多卡推理
- **预计显存**: 约 16-20GB per GPU
- **预计时间**: 数小时
- **性能损失**: 无

---

## 5. 具体复现方案

### 5.1 推理验证方案（推荐）

**目标**: 验证官方模型性能，学习代码实现

**硬件需求**:
- 1 张 4090 (24GB) 即可
- 推荐 2 张 4090 用于 32B 模型

**步骤**:
1. 下载官方 checkpoint:
   ```bash
   # 8B 模型
   huggingface-cli download OpenSearch-VL/OpenSearch-VL-8B
   
   # 30B-A3B 模型
   huggingface-cli download OpenSearch-VL/OpenSearch-VL-30B-A3B
   
   # 32B 模型
   huggingface-cli download OpenSearch-VL/OpenSearch-VL-32B
   ```

2. 配置推理环境:
   ```bash
   cd opensearch_vl
   pip install torch transformers qwen-vl-utils accelerate \
               pandas pyarrow Pillow opencv-python tqdm requests httpx
   ```

3. 运行推理:
   ```bash
   # 8B 模型，单卡
   python run_infer.py --model 8b --gpus 0 \
       --data-path /path/to/benchmark.parquet \
       --output-dir ./outputs/opensearch_vl_8b
   
   # 32B 模型，4 卡
   python run_infer.py --model 32b --gpus 0,1,2,3 \
       --data-path /path/to/benchmark.parquet \
       --output-dir ./outputs/opensearch_vl_32b
   ```

4. 评估结果:
   ```bash
   python eval_with_gpt4o.py \
       --traj_dir ./outputs/opensearch_vl_8b/bc_vl_level1 \
       --benchmark bc_vl \
       --max_workers 20
   ```

**优点**:
- 资源需求最低
- 可以验证论文结果
- 学习完整的推理流程
- 理解工具使用和多轮对话

### 5.2 LoRA 微调方案

**目标**: 使用 LoRA 微调 8B 模型，学习 SFT 流程

**硬件需求**:
- 8 张 4090 (24GB)
- 总显存: 192GB

**配置调整**:

```yaml
# 修改 SFT 配置 (qwen3_vl_full_sft_8b_ray.yaml)
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
cutoff_len: 8192  # 减少序列长度
num_train_epochs: 3
learning_rate: 2e-4
bf16: true
gradient_checkpointing: true
deepspeed: examples/deepspeed/ds_z3_offload_config.json  # 使用 CPU Offload
```

**DeepSpeed 配置** (ds_z3_offload_config.json):
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**启动命令**:
```bash
cd SFT
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
llamafactory-cli train examples/agentic_full/qwen3_vl_full_sft_8b_lora.yaml
```

**预计资源使用**:
- 每卡显存: 约 18-20GB
- 训练时间: 约 1-2 周
- 性能损失: 约 5-10%

### 5.3 单节点 RL 实验方案

**目标**: 在单节点上运行 RL 训练的简化版本

**硬件需求**:
- 8 张 4090 (24GB)
- 总显存: 192GB

**前提条件**:
- 使用 LoRA 微调后的 8B 模型
- 配置 SGLang 推理引擎

**配置调整**:

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
train_prompt_mini_bsz=8
max_response_length=10000
gpu_memory_utilization=0.7
offload=True
```

**启动命令**:
```bash
cd RL/rllm
bash vision_deepresearch_async_workflow/run/qwen3-vl-8b-single-node.sh
```

**预计资源使用**:
- 每卡显存: 约 20-22GB
- 训练时间: 约 1 周
- 性能损失: 显著（约 20-30%）

**注意事项**:
- 需要配置搜索 API (Serper/Jina)
- 需要配置 GPT-4o API 用于奖励计算
- 训练可能不稳定，需要仔细监控

---

## 6. 资源优化技巧

### 6.1 显存优化

1. **Gradient Checkpointing**: 用计算换显存，减少激活值存储
2. **CPU Offload**: 将优化器状态和梯度卸载到 CPU
3. **混合精度训练**: 使用 bfloat16 减少显存占用
4. **减少序列长度**: 从 32000 减少到 8192 或更小
5. **减少批量大小**: 使用 gradient accumulation
6. **LoRA/QLoRA**: 减少可训练参数

### 6.2 计算优化

1. **Flash Attention**: 加速注意力计算，减少显存占用
2. **Tensor Parallelism**: 将矩阵乘法分片到多个 GPU
3. **Pipeline Parallelism**: 将模型层分到多个阶段
4. **异步训练**: 使用 SGLang 进行异步 rollout

### 6.3 通信优化

1. **NVLink**: 消费级 GPU 缺乏 NVLink，多卡通信效率低
2. **InfiniBand**: 集群环境使用 InfiniBand 进行高速通信
3. **梯度压缩**: 减少通信量
4. **异步通信**: 重叠计算和通信

---

## 7. 总结

### 关键发现

1. **官方资源需求极高**: SFT 需要 256 张 H20 GPU，RL 需要 64 张 H100 GPU
2. **用户资源严重不足**: 16 张 24GB GPU 无法直接复现论文结果
3. **可行的复现方案**: 推理验证、LoRA 微调、单节点 RL 实验

### 推荐路径

1. **第一步**: 推理验证，使用官方 checkpoint 验证性能
2. **第二步**: LoRA 微调 8B 模型，学习 SFT 流程
3. **第三步**: 单节点 RL 实验，理解 RL 训练流程
4. **第四步**: 应用到自己的项目，借鉴创新点

### 资源需求总结

| 方案 | GPU 需求 | 显存需求 | 训练时间 | 性能损失 |
|------|----------|----------|----------|----------|
| 推理验证 | 1-4 张 4090 | 16-20GB | 数小时 | 无 |
| LoRA 微调 | 8 张 4090 | 18-20GB | 1-2 周 | 5-10% |
| 单节点 RL | 8 张 4090 | 20-22GB | 1 周 | 20-30% |
| 官方 SFT | 256 张 H20 | 96GB | 2-4 天 | 0% |
| 官方 RL | 64 张 H100 | 96GB | 10 天 | 0% |

---

## 参考资料

- **项目文档**: `/home/chenyizhou/OpenSearch-VL/README.md`
- **SFT 文档**: `/home/chenyizhou/OpenSearch-VL/SFT/README.md`
- **RL 文档**: `/home/chenyizhou/OpenSearch-VL/RL/README.md`
- **内存估算工具**: `/home/chenyizhou/OpenSearch-VL/RL/mbridge/memory_estimator/`
- **训练配置**: `/home/chenyizhou/OpenSearch-VL/SFT/examples/agentic_full/`
- **RL 启动脚本**: `/home/chenyizhou/OpenSearch-VL/RL/rllm/vision_deepresearch_async_workflow/run/`
