# OpenSearch-VL 8×RTX 3090 完整复现计划

## 一、资源评估与可行性分析

### 1.1 硬件资源对比

| 对比项 | 官方资源 | 用户资源 | 差距 |
|--------|----------|----------|------|
| **GPU型号** | H20/H100 | RTX 3090 | — |
| **单卡显存** | 96GB | 24GB | **4×** |
| **GPU数量** | 256 (SFT) / 64 (RL) | 8 | **32× / 8×** |
| **总显存** | 24,576GB (SFT) | 192GB | **128×** |
| **FP16算力** | 989 TFLOPS | 35.6 TFLOPS | **28×** |
| **显存带宽** | 3,352 GB/s | 936 GB/s | **3.6×** |
| **NVLink** | 支持 | 不支持 | — |

### 1.2 核心挑战

1. **显存不足**：24GB无法容纳8B模型的优化器状态（需要96GB）
2. **算力不足**：训练时间将是官方的28倍以上
3. **通信瓶颈**：消费级GPU缺乏NVLink，多卡通信效率低

### 1.3 可行性结论

| 复现阶段 | 可行性 | 方案 | 预期效果 |
|----------|--------|------|----------|
| **推理验证** | ✅ 完全可行 | 单卡/多卡推理 | 无性能损失 |
| **SFT训练** | ⚠️ 有限可行 | LoRA + CPU Offload | 性能损失5-10% |
| **RL训练** | ⚠️ 有限可行 | 极致优化 + 缩减配置 | 性能损失20-30% |
| **完整流程** | ⚠️ 需要调整 | 使用3B模型或更激进优化 | 性能损失30-50% |

---

## 二、完整复现计划

### Phase 0: 环境准备（1天）

#### 2.0.1 硬件检查
```bash
# 检查GPU状态
nvidia-smi

# 检查GPU互联
nvidia-smi topo -m

# 检查系统内存（需要大内存用于CPU Offload）
free -h

# 检查磁盘空间
df -h
```

**需求**：
- 8张RTX 3090，每张24GB
- 系统内存：至少256GB（用于CPU Offload）
- 磁盘空间：至少500GB（模型+数据+checkpoint）

#### 2.0.2 环境安装
```bash
# 1. 创建conda环境
conda create -n opensearch python=3.10 -y
conda activate opensearch

# 2. 安装PyTorch
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装SFT依赖
cd /home/chenyizhou/OpenSearch-VL/SFT
pip install -e ".[torch,metrics,deepspeed,ray]"
pip install qwen-vl-utils pillow av decord torchvision flash-attn

# 4. 安装RL依赖
cd ../RL/rllm && pip install -e .
cd ../Megatron-LM && pip install -e .
cd ../mbridge && pip install -e .
pip install "sglang[all]" transformer_engine flash-attn ray==2.34.* hydra-core omegaconf wandb

# 5. 安装推理依赖
cd ../../opensearch_vl
pip install torch transformers qwen-vl-utils accelerate pandas pyarrow Pillow opencv-python
```

---

### Phase 1: 推理验证（2-3天）

#### 2.1.1 目标
- 验证官方模型性能
- 理解推理流程
- 测试工具环境

#### 2.1.2 步骤

**Step 1: 下载模型和数据**
```bash
# 下载8B模型（约16GB）
huggingface-cli download OpenSearch-VL/OpenSearch-VL-8B --local-dir /data/models/opensearch-vl-8b

# 下载评估数据
huggingface-cli download OpenSearch-VL/SearchVL-SFT-36k --local-dir /data/datasets/sft-36k
```

**Step 2: 配置环境变量**
```bash
# 复制环境变量模板
cp opensearch_vl/.env.example ~/.opensearch-vl.env

# 编辑环境变量
vim ~/.opensearch-vl.env
# 填入：
# - QWEN3VL_8B_PATH=/data/models/opensearch-vl-8b
# - SERPER_API_KEY=xxx（可选，用于搜索工具）
# - JINA_API_KEY=xxx（可选，用于页面访问）

# 加载环境变量
source ~/.opensearch-vl.env
```

**Step 3: 单卡推理测试**
```bash
cd /home/chenyizhou/OpenSearch-VL

# 单卡推理（8B模型约需要18GB显存）
python opensearch_vl/run_infer.py --model 8b --gpus 0 \
    --data-path /data/datasets/eval/benchmark.parquet \
    --output-dir ./outputs/opensearch_vl_8b \
    --start 0 --end 100
```

**Step 4: 多卡推理测试**
```bash
# 4卡推理（用于更大的batch或更长的序列）
python opensearch_vl/run_infer.py --model 8b --gpus 0,1,2,3 \
    --data-path /data/datasets/eval/benchmark.parquet \
    --output-dir ./outputs/opensearch_vl_8b_4gpu \
    --start 0 --end 500
```

**Step 5: 评估结果**
```bash
# 使用GPT-4o评估（需要API Key）
python opensearch_vl/eval_with_gpt4o.py \
    --traj_dir ./outputs/opensearch_vl_8b/bc_vl_level1 \
    --benchmark bc_vl \
    --max_workers 20
```

#### 2.1.3 预期结果
- 单卡推理：18GB显存，每条约30-60秒
- 4卡推理：每条约10-20秒
- 性能：与官方一致（无损失）

---

### Phase 2: SFT训练（1-2周）

#### 2.2.1 目标
- 使用LoRA微调8B模型
- 学习SFT训练流程
- 验证训练效果

#### 2.2.2 方案选择

**方案A：LoRA微调（推荐）**
- 显存需求：约18-20GB/卡
- 性能损失：5-10%
- 训练时间：约1-2周

**方案B：QLoRA微调**
- 显存需求：约12-15GB/卡
- 性能损失：10-15%
- 训练时间：约1-2周

**方案C：全参数微调（极致优化）**
- 显存需求：约22-24GB/卡（非常紧张）
- 性能损失：0-5%
- 训练时间：约2-3周
- 风险：可能OOM

#### 2.2.3 LoRA微调详细步骤

**Step 1: 准备数据**
```bash
# 下载SFT数据
huggingface-cli download OpenSearch-VL/SearchVL-SFT-36k --local-dir /data/datasets/sft-36k

# 创建数据目录
mkdir -p SFT/data/new_fvqa SFT/data/palace SFT/data/WebQA
mkdir -p SFT/data/new_livevqa SFT/data/wikiart SFT/data/wiki_en SFT/data/wiki_zh

# 复制数据（根据dataset_info.json中的路径）
cp /data/datasets/sft-36k/fvqa_llama_factory_clean.json SFT/data/new_fvqa/
cp /data/datasets/sft-36k/palace_llama_factory_filtered.json SFT/data/palace/
# ... 其他数据类似
```

**Step 2: 创建LoRA配置文件**
```yaml
# SFT/examples/agentic_full/qwen3_vl_lora_8b_3090.yaml

### model
model_name_or_path: /data/models/opensearch-vl-8b
image_max_pixels: 131072  # 从262144减少，降低显存
video_max_pixels: 8192    # 从16384减少
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
freeze_vision_tower: false
freeze_multi_modal_projector: false
deepspeed: examples/deepspeed/ds_z3_offload_config.json

### dataset
dataset: new_fvqa_agent_sft,palace_agent_sft,webqa_agent_sft,livevqa_agent_sft,wikiart_agent_sft,wiki_zh_agent_sft,wiki_en_agent_sft
dataset_dir: data
template: qwen3_vl
cutoff_len: 8192  # 从32000减少，大幅降低显存
overwrite_cache: true
preprocessing_num_workers: 8
dataloader_num_workers: 2

### output
output_dir: saves/qwen3_vl_8b/lora/sft_3090
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 有效batch_size = 1 * 16 * 8 = 128
gradient_checkpointing: true
learning_rate: 2e-4  # LoRA通常用更大的学习率
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 600
resume_from_checkpoint: true

### 3090特定优化
optim: adamw_torch  # 使用PyTorch原生优化器
max_grad_norm: 1.0
```

**Step 3: 创建DeepSpeed Offload配置**
```json
// SFT/examples/deepspeed/ds_z3_offload_3090.json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true,
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

**Step 4: 启动训练**
```bash
cd /home/chenyizhou/OpenSearch-VL/SFT

# 单节点8卡训练
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=8 \
llamafactory-cli train examples/agentic_full/qwen3_vl_lora_8b_3090.yaml

# 或者使用Ray（如果配置正确）
USE_RAY=1 llamafactory-cli train examples/agentic_full/qwen3_vl_lora_8b_3090.yaml
```

**Step 5: 监控训练**
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控训练日志
tail -f saves/qwen3_vl_8b/lora/sft_3090/trainer_log.jsonl

# 启动TensorBoard
tensorboard --logdir saves/qwen3_vl_8b/lora/sft_3090 --port 6006
```

**Step 6: 合并LoRA权重**
```bash
# 训练完成后，合并LoRA权重到基础模型
llamafactory-cli export \
    --model_name_or_path /data/models/opensearch-vl-8b \
    --adapter_name_or_path saves/qwen3_vl_8b/lora/sft_3090 \
    --template qwen3_vl \
    --finetuning_type lora \
    --export_dir /data/models/opensearch-vl-8b-sft-lora
```

#### 2.2.4 预期结果
- 显存使用：约18-20GB/卡
- 训练时间：约1-2周（3 epochs）
- 性能损失：5-10%

---

### Phase 3: RL训练（1-2周）

#### 2.3.1 目标
- 使用Fatal-Aware GRPO训练
- 学习RL训练流程
- 验证算法效果

#### 2.3.2 方案选择

**方案A：使用LoRA微调后的8B模型（推荐）**
- 基础模型：Phase 2的LoRA微调模型
- 显存需求：约20-22GB/卡
- 性能损失：20-30%
- 训练时间：约1-2周

**方案B：使用更小的3B模型**
- 基础模型：Qwen2.5-3B-Instruct
- 显存需求：约12-15GB/卡
- 性能损失：30-50%
- 训练时间：约1周

#### 2.3.3 RL训练详细步骤

**Step 1: 准备RL数据**
```bash
cd /home/chenyizhou/OpenSearch-VL/RL/rllm/vision_deepresearch_async_workflow/data_prepare

# 下载RL数据
huggingface-cli download OpenSearch-VL/SearchVL-RL-8k --local-dir ./data/Vision-DeepResearch-RL-Data

# 转换数据格式
DATA_ROOT=./data/Vision-DeepResearch-RL-Data bash convert_parquet2jsonl.sh

# 注册数据集
JSONL_PATH=./data/Vision-DeepResearch-RL-Data/vision-deepresearch_RL_Demo_1k.jsonl \
    bash register_rl_dataset.sh
```

**Step 2: 配置环境变量**
```bash
cd /home/chenyizhou/OpenSearch-VL/RL/rllm

# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
vim .env
# 填入：
# - WANDB_API_KEY=xxx（可选，用于日志）
# - SERPER_API_KEY=xxx（用于搜索工具）
# - JINA_API_KEY=xxx（用于页面访问）
# - JUDGE_API_BASE_URL=xxx（用于奖励计算）
# - JUDGE_API_KEY=xxx
```

**Step 3: 创建单节点训练脚本**
```bash
#!/bin/bash
# RL/rllm/vision_deepresearch_async_workflow/run/qwen3-vl-8b-3090.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# 加载环境变量
if [ -f .env ]; then
  set -a; source .env; set +a
fi

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

dtype="bfloat16"
adv_estimator="rloo"

# KL/Clip参数
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_high=0.28

# 批量大小（针对3090优化）
train_prompt_bsz=64      # 从256减少
n_resp_per_prompt=4      # 从8减少
train_prompt_mini_bsz=16 # 从64减少
n_parallel_tasks=64      # 从256减少
n_parallel_tools=512     # 从2048减少

# 序列长度
max_prompt_length=2048   # 从4096减少
max_response_length=16384 # 从70000减少

use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=18432  # 从74576减少
infer_ppo_max_token_len_per_gpu=18432

# 并行策略（针对3090优化）
offload=True
gen_tp=2                 # 从4减少
train_tp=2
train_pp=1
train_cp=1

# 采样参数
temperature=0.7
top_p=1.0
top_k=-1
val_top_p=0.95

loss_agg_mode="seq-mean-token-sum"

# 集群配置
NNODES=1
project_name='vision-deepresearch'
exp_name='open_mm_searcher_8b_3090'

# 模型路径（使用LoRA微调后的模型）
MODEL_PATH=/data/models/opensearch-vl-8b-sft-lora
CKPTS_DIR=checkpoints/${project_name}/${exp_name}

python3 -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=3407 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.megatron.dtype=${dtype} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.megatron.dtype=${dtype} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${offload} \
    \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=25 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}"
```

**Step 4: 启动RL训练**
```bash
cd /home/chenyizhou/OpenSearch-VL/RL/rllm

# 启动训练
bash vision_deepresearch_async_workflow/run/qwen3-vl-8b-3090.sh
```

**Step 5: 监控训练**
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控训练日志
tail -f checkpoints/vision-deepresearch/open_mm_searcher_8b_3090/*.log

# 监控W&B（如果配置）
# 访问 https://wandb.ai 查看训练曲线
```

#### 2.3.4 预期结果
- 显存使用：约20-22GB/卡
- 训练时间：约1-2周（10 epochs）
- 性能损失：20-30%

---

### Phase 4: 评估与验证（2-3天）

#### 2.4.1 目标
- 评估训练后的模型
- 与官方结果对比
- 分析性能差距

#### 2.4.2 步骤

**Step 1: 使用训练后的模型推理**
```bash
cd /home/chenyizhou/OpenSearch-VL

# 使用RL训练后的模型
python opensearch_vl/run_infer.py --model 8b --gpus 0,1,2,3 \
    --checkpoint /data/models/opensearch-vl-8b-rl \
    --data-path /data/datasets/eval/benchmark.parquet \
    --output-dir ./outputs/opensearch_vl_8b_rl \
    --start 0 --end 1000
```

**Step 2: 评估结果**
```bash
# 评估各个基准
python opensearch_vl/eval_with_gpt4o.py \
    --traj_dir ./outputs/opensearch_vl_8b_rl/bc_vl_level1 \
    --benchmark bc_vl \
    --max_workers 20

python opensearch_vl/eval_with_gpt4o.py \
    --traj_dir ./outputs/opensearch_vl_8b_rl/hle \
    --benchmark hle \
    --max_workers 20

python opensearch_vl/eval_with_gpt4o.py \
    --traj_dir ./outputs/opensearch_vl_8b_rl/vdr \
    --benchmark vdr \
    --max_workers 20
```

**Step 3: 对比分析**
```bash
# 创建对比表格
python scripts/compare_results.py \
    --official_dir ./results/official \
    --reproduced_dir ./outputs/opensearch_vl_8b_rl \
    --output comparison.md
```

---

## 三、时间规划

### 3.1 总体时间线

| 阶段 | 任务 | 预计时间 | 累计时间 |
|------|------|----------|----------|
| Phase 0 | 环境准备 | 1天 | 1天 |
| Phase 1 | 推理验证 | 2-3天 | 3-4天 |
| Phase 2 | SFT训练 | 1-2周 | 10-18天 |
| Phase 3 | RL训练 | 1-2周 | 17-32天 |
| Phase 4 | 评估验证 | 2-3天 | 19-35天 |
| **总计** | — | **3-5周** | — |

### 3.2 详细时间表

**第1周**：
- Day 1: 环境准备
- Day 2-4: 推理验证
- Day 5-7: SFT数据准备和配置

**第2-3周**：
- SFT训练（可能需要多次调整超参数）

**第3-4周**：
- RL数据准备
- RL训练

**第5周**：
- 评估验证
- 结果分析

---

## 四、风险与应对

### 4.1 潜在风险

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|----------|
| OOM（显存不足） | 高 | 训练失败 | 减小batch size、增加gradient accumulation、使用更激进的offload |
| 训练不收敛 | 中 | 结果差 | 调整学习率、增加SFT数据、调整K值 |
| 工具API不稳定 | 中 | RL训练失败 | 增加重试机制、使用本地工具替代 |
| 时间超预期 | 中 | 延期 | 减少训练epochs、使用更小的模型 |

### 4.2 应急方案

**如果OOM**：
```bash
# 方案1：减小batch size
train_prompt_bsz=32  # 从64减少
train_prompt_mini_bsz=8

# 方案2：增加gradient accumulation
gradient_accumulation_steps=32  # 从16增加

# 方案3：使用QLoRA
finetuning_type: qlora
quantization_bit: 4
```

**如果训练不收敛**：
```bash
# 方案1：降低学习率
learning_rate=1e-4  # 从2e-4降低

# 方案2：增加SFT数据
# 使用更多epochs或更多数据

# 方案3：调整K值
_CONSECUTIVE_ERROR_THRESHOLD=5  # 从3增加
```

**如果时间超预期**：
```bash
# 方案1：减少训练epochs
num_train_epochs=1  # 从3减少
trainer.total_epochs=5  # 从10减少

# 方案2：使用更小的模型
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
```

---

## 五、关键配置参数

### 5.1 SFT配置参数

| 参数 | 官方值 | 3090值 | 说明 |
|------|--------|--------|------|
| `cutoff_len` | 32000 | 8192 | 序列长度 |
| `per_device_train_batch_size` | 1 | 1 | 每卡batch size |
| `gradient_accumulation_steps` | 1 | 16 | 梯度累积 |
| `learning_rate` | 2e-5 | 2e-4 | LoRA学习率 |
| `num_train_epochs` | 8 | 3 | 训练轮数 |
| `lora_rank` | — | 16 | LoRA秩 |
| `lora_alpha` | — | 32 | LoRA alpha |

### 5.2 RL配置参数

| 参数 | 官方值 | 3090值 | 说明 |
|------|--------|--------|------|
| `train_prompt_bsz` | 256 | 64 | 训练batch size |
| `n_resp_per_prompt` | 8 | 4 | 每个prompt的响应数 |
| `train_prompt_mini_bsz` | 64 | 16 | mini-batch size |
| `max_prompt_length` | 4096 | 2048 | prompt最大长度 |
| `max_response_length` | 70000 | 16384 | 响应最大长度 |
| `gen_tp` | 4 | 2 | 推理TP |
| `train_tp` | 4 | 2 | 训练TP |
| `gpu_memory_utilization` | 0.85 | 0.7 | GPU显存利用率 |
| `trainer.total_epochs` | 100 | 10 | 训练轮数 |

---

## 六、监控与调试

### 6.1 关键监控指标

| 指标 | 正常范围 | 异常处理 |
|------|----------|----------|
| GPU显存使用 | <22GB | 减小batch size |
| GPU利用率 | >80% | 检查数据加载 |
| 训练loss | 下降趋势 | 检查学习率 |
| 奖励值 | 上升趋势 | 检查奖励函数 |
| 致命轨迹比例 | <30% | 调整K值 |

### 6.2 调试命令

```bash
# 检查GPU状态
nvidia-smi

# 检查进程
ps aux | grep python

# 检查内存使用
free -h

# 检查磁盘空间
df -h

# 查看训练日志
tail -f checkpoints/*/trainer_log.jsonl

# 查看TensorBoard
tensorboard --logdir checkpoints/ --port 6006
```

---

## 七、学习检查清单

### 7.1 Phase 0 完成标准
- [ ] 环境安装成功
- [ ] GPU状态正常
- [ ] 数据下载完成

### 7.2 Phase 1 完成标准
- [ ] 单卡推理成功
- [ ] 多卡推理成功
- [ ] 评估结果与官方一致

### 7.3 Phase 2 完成标准
- [ ] SFT训练完成
- [ ] loss收敛
- [ ] LoRA权重合并成功

### 7.4 Phase 3 完成标准
- [ ] RL训练完成
- [ ] 奖励值上升
- [ ] 致命轨迹比例下降

### 7.5 Phase 4 完成标准
- [ ] 评估完成
- [ ] 结果对比完成
- [ ] 性能差距分析完成

---

## 八、参考资源

### 8.1 项目文档
- **README**: /home/chenyizhou/OpenSearch-VL/README.md
- **SFT文档**: /home/chenyizhou/OpenSearch-VL/SFT/README.md
- **RL文档**: /home/chenyizhou/OpenSearch-VL/RL/README.md
- **学习笔记**: /home/chenyizhou/OpenSearch-VL/doc/

### 8.2 外部资源
- **论文**: https://arxiv.org/pdf/2605.05185
- **GitHub**: https://github.com/shawn0728/OpenSearch-VL
- **HuggingFace**: https://huggingface.co/OpenSearch-VL

### 8.3 技术栈文档
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **rLLM**: https://github.com/rllm-org/rllm
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM
- **DeepSpeed**: https://github.com/microsoft/DeepSpeed

---

## 总结

本计划针对8×RTX 3090的硬件资源，设计了完整的OpenSearch-VL复现方案。通过LoRA微调、CPU Offload、序列长度缩减等技术，在性能损失可控的情况下完成全流程复现。

**关键成功因素**：
1. **显存优化**：LoRA + CPU Offload + 序列长度缩减
2. **时间管理**：预留足够的时间缓冲
3. **风险应对**：准备好应急方案
4. **持续监控**：及时发现和解决问题

**预期成果**：
- SFT性能损失：5-10%
- RL性能损失：20-30%
- 总训练时间：3-5周
