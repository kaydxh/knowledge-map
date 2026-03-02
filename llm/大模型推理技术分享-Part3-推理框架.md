# 大模型推理技术分享 - Part 3: 推理框架对比与vLLM深入解析

> **受众**: AI平台开发工程师
> **目标**: 理解主流推理框架的设计差异，深入掌握vLLM

---

## 目录

1. [推理框架概览](#1-推理框架概览)
2. [vLLM vs SGLang vs TensorRT-LLM 对比](#2-vllm-vs-sglang-vs-tensorrt-llm-对比)
3. [vLLM 核心启动参数](#3-vllm-核心启动参数)
4. [vLLM 关键设计](#4-vllm-关键设计)
5. [生产环境最佳实践](#5-生产环境最佳实践)

---

## 1. 推理框架概览

### 1.1 主流框架对比

| 框架 | 开发者 | 核心特性 | 适用场景 |
|------|--------|---------|---------|
| **vLLM** | UC Berkeley | PagedAttention, Continuous Batching | 通用推理，生产环境 |
| **SGLang** | LMSYS | RadixAttention, PD分离 | Few-shot, 对话, Agent |
| **TensorRT-LLM** | NVIDIA | 深度编译优化，FP8 | 极致性能，NVIDIA GPU |
| **Text Generation Inference** | HuggingFace | 易用性，社区支持 | 快速部署，原型验证 |
| **LMDeploy** | OpenMMLab | 多模态，量化 | VLM推理，中文优化 |

### 1.2 技术栈对比

```
┌─────────────────────────────────────────────────────────┐
│                    应用层                                │
│  OpenAI API, LangChain, LlamaIndex                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  推理框架层                              │
│  vLLM, SGLang, TensorRT-LLM, TGI                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  算子库层                                │
│  FlashAttention, xFormers, cuBLAS, CUTLASS             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  硬件层                                  │
│  CUDA, ROCm, Neuron                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. vLLM vs SGLang vs TensorRT-LLM 对比

### 2.1 架构对比

#### vLLM 架构
```
┌─────────────────────────────────────────────────────────┐
│                   API Server (FastAPI)                   │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   LLMEngine                              │
│  - 请求管理                                              │
│  - 调度协调                                              │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   Scheduler                              │
│  - Continuous Batching                                   │
│  - BlockSpaceManager (PagedAttention)                   │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   Worker (GPU)                           │
│  - ModelRunner                                           │
│  - CacheEngine                                           │
└─────────────────────────────────────────────────────────┘
```

**特点**：
- 单进程架构（简单）
- PagedAttention（块级KV Cache管理）
- 成熟稳定，社区活跃

#### SGLang 架构
```
┌─────────────────────────────────────────────────────────┐
│                   HTTP Server (FastAPI)                  │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                TokenizerManager (主进程)                 │
│  - Tokenization                                          │
│  - Chat Template                                         │
└────────────────────────┬────────────────────────────────┘
                         │ ZMQ IPC
                         ↓
┌─────────────────────────────────────────────────────────┐
│                Scheduler (子进程, GPU)                   │
│  - RadixCache (前缀树)                                   │
│  - LPF调度策略                                           │
│  - ModelRunner                                           │
└────────────────────────┬────────────────────────────────┘
                         │ ZMQ IPC
                         ↓
┌─────────────────────────────────────────────────────────┐
│              DetokenizerManager (子进程)                 │
│  - Detokenization                                        │
│  - 流式输出                                              │
└─────────────────────────────────────────────────────────┘
```

**特点**：
- 三进程架构（解耦）
- RadixAttention（前缀树KV Cache管理）
- 适合Few-shot、对话、Agent场景

#### TensorRT-LLM 架构
```
┌─────────────────────────────────────────────────────────┐
│                   Triton Server                          │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                TensorRT-LLM Runtime                      │
│  - 编译优化的Engine                                      │
│  - Inflight Batching                                     │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                TensorRT Engine                           │
│  - Kernel Fusion                                         │
│  - FP8/INT8 量化                                         │
│  - CUDA Graph                                            │
└─────────────────────────────────────────────────────────┘
```

**特点**：
- 深度编译优化
- 极致性能（但灵活性较低）
- NVIDIA官方支持

### 2.2 KV Cache 管理对比

#### vLLM: PagedAttention
```python
# 块级管理
class BlockSpaceManager:
    def __init__(self, block_size=16, num_blocks=1000):
        self.block_size = 16  # 固定块大小
        self.block_tables = {}  # {seq_id: [block_ids]}
    
    def allocate(self, seq_id, num_tokens):
        num_blocks = (num_tokens + 15) // 16
        blocks = self.allocate_blocks(num_blocks)
        self.block_tables[seq_id] = blocks
```

**优势**：
- 减少内存碎片
- 支持块共享（前缀共享）
- 实现简单

**劣势**：
- 固定块大小（可能浪费）
- 前缀共享需要显式指定

#### SGLang: RadixAttention
```python
# 前缀树管理
class RadixCache:
    def __init__(self):
        self.root = TreeNode()
    
    def match_prefix(self, token_ids):
        """自动匹配最长公共前缀"""
        node = self.root
        matched_len = 0
        for token_id in token_ids:
            if token_id in node.children:
                node = node.children[token_id]
                matched_len += 1
            else:
                break
        return matched_len
```

**优势**：
- 自动前缀识别
- 任意长度前缀
- 适合Few-shot场景

**劣势**：
- 实现复杂
- 树维护开销

#### TensorRT-LLM: Inflight Batching
```cpp
// 动态批处理 + KV Cache池
class InflightBatching {
    // 预分配KV Cache池
    std::vector<KVCacheBlock> kv_cache_pool;
    
    // 动态分配
    void allocate(int seq_id, int num_tokens) {
        // 从池中分配
    }
};
```

**优势**：
- 高性能（编译优化）
- 低延迟

**劣势**：
- 灵活性较低
- 需要预编译

### 2.3 性能对比

**测试环境**：
- 模型：Llama-2-7B
- GPU：A100 (80GB)
- Batch Size：32
- Input Length：512
- Output Length：128

| 框架 | 吞吐量 (tokens/s) | TTFT (ms) | TPOT (ms) | 内存占用 (GB) |
|------|------------------|-----------|-----------|--------------|
| **vLLM** | 2800 | 45 | 12 | 28 |
| **SGLang** | 3200 | 42 | 11 | 26 |
| **TensorRT-LLM** | 3800 | 35 | 9 | 24 |
| **TGI** | 2400 | 50 | 14 | 30 |

**结论**：
- **TensorRT-LLM**：性能最高，但灵活性最低
- **SGLang**：Few-shot场景下性能最优
- **vLLM**：通用场景下最佳选择
- **TGI**：易用性最好，性能略低

### 2.4 功能对比

| 功能 | vLLM | SGLang | TensorRT-LLM | TGI |
|------|------|--------|--------------|-----|
| **Continuous Batching** | ✅ | ✅ | ✅ | ✅ |
| **PagedAttention** | ✅ | ❌ | ❌ | ✅ |
| **RadixAttention** | ❌ | ✅ | ❌ | ❌ |
| **Speculative Decoding** | ✅ | ✅ | ✅ | ❌ |
| **LoRA** | ✅ | ✅ | ✅ | ✅ |
| **多模态** | ✅ | ✅ | ✅ | ✅ |
| **结构化输出** | ✅ | ✅ | ❌ | ❌ |
| **FP8** | ✅ | ✅ | ✅ | ❌ |
| **INT8** | ✅ | ✅ | ✅ | ✅ |
| **CUDA Graph** | ✅ | ✅ | ✅ | ❌ |
| **Tensor Parallelism** | ✅ | ✅ | ✅ | ✅ |
| **Pipeline Parallelism** | ✅ | ✅ | ✅ | ❌ |

---

## 3. vLLM 核心启动参数

### 3.1 基础参数

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

### 3.2 模型相关参数

#### `--model` (必需)
```bash
--model meta-llama/Llama-2-7b-chat-hf
```
- HuggingFace模型路径或本地路径
- 支持的模型：Llama, Mistral, Qwen, GPT-NeoX等

#### `--tokenizer`
```bash
--tokenizer meta-llama/Llama-2-7b-chat-hf
```
- 指定tokenizer路径（默认与model相同）
- 用于自定义tokenizer

#### `--revision`
```bash
--revision main
```
- 指定模型版本（Git分支/tag）

#### `--tokenizer-mode`
```bash
--tokenizer-mode auto  # auto, slow
```
- `auto`: 优先使用fast tokenizer
- `slow`: 使用slow tokenizer

### 3.3 内存相关参数

#### `--gpu-memory-utilization`
```bash
--gpu-memory-utilization 0.9  # 默认0.9
```
- GPU显存使用比例（0.0-1.0）
- **关键参数**：影响KV Cache大小
- 建议：0.85-0.95

**计算公式**：
```
KV Cache Size = (Total GPU Memory - Model Weights) × gpu_memory_utilization
```

**示例**（Llama-2-7B, A100 80GB）：
```
Model Weights: 14 GB (FP16)
Available Memory: 80 - 14 = 66 GB
KV Cache (0.9): 66 × 0.9 = 59.4 GB
Max Concurrent Requests: 59.4 / 1 GB ≈ 59
```

#### `--max-model-len`
```bash
--max-model-len 4096  # 默认模型配置
```
- 最大序列长度
- 影响KV Cache分配
- 建议：根据实际需求设置

#### `--block-size`
```bash
--block-size 16  # 默认16
```
- PagedAttention块大小
- 必须是2的幂（8, 16, 32）
- 影响内存碎片和性能

**选择建议**：
- 短序列（<512）：block_size=8
- 中等序列（512-2048）：block_size=16
- 长序列（>2048）：block_size=32

### 3.4 并行相关参数

#### `--tensor-parallel-size`
```bash
--tensor-parallel-size 4  # 默认1
```
- Tensor并行度
- 必须是GPU数量的因子
- 适合：单节点多GPU

**示例**（8×A100）：
```bash
# TP=8（单个模型副本）
--tensor-parallel-size 8

# TP=4, DP=2（2个模型副本）
--tensor-parallel-size 4
```

#### `--pipeline-parallel-size`
```bash
--pipeline-parallel-size 2  # 默认1
```
- Pipeline并行度
- 适合：跨节点部署
- 注意：会增加延迟

#### `--distributed-executor-backend`
```bash
--distributed-executor-backend ray  # ray, mp
```
- `ray`: 使用Ray（推荐，跨节点）
- `mp`: 使用multiprocessing（单节点）

### 3.5 性能相关参数

#### `--max-num-seqs`
```bash
--max-num-seqs 256  # 默认256
```
- 最大并发序列数
- **关键参数**：影响吞吐量
- 建议：根据GPU内存调整

**计算公式**：
```
max_num_seqs = KV Cache Size / (max_model_len × kv_cache_per_token)
```

#### `--max-num-batched-tokens`
```bash
--max-num-batched-tokens 8192  # 默认根据模型计算
```
- 单个batch的最大token数
- 影响Prefill阶段性能
- 建议：2048-8192

#### `--enable-prefix-caching`
```bash
--enable-prefix-caching
```
- 启用前缀缓存（类似RadixAttention）
- 适合：Few-shot场景
- **推荐开启**

#### `--disable-log-stats`
```bash
--disable-log-stats
```
- 禁用统计日志
- 减少CPU开销
- 生产环境建议开启

### 3.6 量化相关参数

#### `--quantization`
```bash
--quantization awq  # awq, gptq, squeezellm, fp8
```
- 量化方法
- `awq`: 推荐，性能好
- `gptq`: 兼容性好
- `fp8`: H100专用

#### `--load-format`
```bash
--load-format auto  # auto, pt, safetensors, npcache, dummy
```
- 模型加载格式
- `safetensors`: 推荐，安全快速
- `npcache`: 预处理缓存，加速启动

### 3.7 调度相关参数

#### `--scheduler-policy`
```bash
--scheduler-policy fcfs  # fcfs, priority
```
- `fcfs`: First-Come-First-Serve
- `priority`: 优先级调度

#### `--max-paddings`
```bash
--max-paddings 256  # 默认256
```
- 最大padding数量
- 影响批处理效率

### 3.8 完整示例

**生产环境配置**（8×A100, Llama-2-70B）：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --block-size 16 \
    --enable-prefix-caching \
    --disable-log-stats \
    --trust-remote-code
```

**Few-shot场景配置**（单A100, Llama-2-7B）：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048 \
    --max-num-seqs 64 \
    --enable-prefix-caching \  # 关键！
    --block-size 16
```

---

## 4. vLLM 关键设计

### 4.1 PagedAttention 实现

**核心文件**：`vllm/core/block_manager.py`

```python
class BlockSpaceManager:
    """管理KV Cache的块分配"""
    
    def __init__(self, block_size: int, num_gpu_blocks: int):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        
        # 空闲块列表
        self.free_blocks = list(range(num_gpu_blocks))
        
        # 块表：{seq_id: [block_ids]}
        self.block_tables = {}
    
    def allocate(self, seq_id: int, num_tokens: int) -> List[int]:
        """为序列分配块"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        
        if len(self.free_blocks) < num_blocks:
            raise ValueError("Out of memory")
        
        # 分配块
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
        
        return blocks
    
    def free(self, seq_id: int):
        """释放序列的块"""
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
    
    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否有足够的块"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= num_blocks
```

**Attention Kernel**：
```python
def paged_attention(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks]
    context_lens: torch.Tensor,  # [num_seqs]
) -> torch.Tensor:
    """
    PagedAttention前向传播
    
    关键：通过block_tables间接访问KV Cache
    """
    # 调用CUDA kernel
    output = _paged_attention_kernel(
        query, key_cache, value_cache, block_tables, context_lens
    )
    return output
```

### 4.2 Continuous Batching 实现

**核心文件**：`vllm/core/scheduler.py`

```python
class Scheduler:
    """调度器：管理请求队列和批处理"""
    
    def __init__(self, scheduler_config, cache_config):
        self.waiting_queue = []  # 等待队列
        self.running_queue = []  # 运行队列
        self.swapped_queue = []  # 换出队列
        
        self.block_manager = BlockSpaceManager(...)
    
    def schedule(self) -> Tuple[SchedulerOutputs, List[SequenceGroup]]:
        """调度主逻辑"""
        
        # 1. 处理换入（Swap In）
        self._schedule_swapped()
        
        # 2. 处理运行中的请求（Decode）
        self._schedule_running()
        
        # 3. 处理新请求（Prefill）
        self._schedule_waiting()
        
        # 4. 构造批处理
        batch = self._create_batch()
        
        return batch
    
    def _schedule_running(self):
        """调度运行中的请求"""
        i = 0
        while i < len(self.running_queue):
            seq_group = self.running_queue[i]
            
            # 检查是否完成
            if seq_group.is_finished():
                self._free_seq_group(seq_group)
                self.running_queue.pop(i)
                continue
            
            # 检查是否需要分配新块
            if not self.block_manager.can_allocate(1):
                # 内存不足，换出
                self._swap_out(seq_group)
                self.running_queue.pop(i)
                self.swapped_queue.append(seq_group)
                continue
            
            # 分配新块（如果需要）
            self.block_manager.append_slot(seq_group)
            i += 1
    
    def _schedule_waiting(self):
        """调度等待中的请求"""
        while self.waiting_queue:
            seq_group = self.waiting_queue[0]
            
            # 检查是否有足够内存
            num_tokens = seq_group.get_num_tokens()
            if not self.block_manager.can_allocate(num_tokens):
                break
            
            # 分配块
            self.block_manager.allocate(seq_group)
            
            # 移到运行队列
            self.waiting_queue.pop(0)
            self.running_queue.append(seq_group)
```

### 4.3 ModelRunner 实现

**核心文件**：`vllm/worker/model_runner.py`

```python
class ModelRunner:
    """模型执行器"""
    
    def __init__(self, model_config, parallel_config, ...):
        # 加载模型
        self.model = self.load_model()
        
        # 初始化KV Cache
        self.cache_engine = CacheEngine(...)
    
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> SamplerOutput:
        """执行模型前向传播"""
        
        # 1. 准备输入
        input_tokens, input_positions, ... = self._prepare_inputs(
            seq_group_metadata_list
        )
        
        # 2. 前向传播
        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.cache_engine.gpu_cache,
            ...
        )
        
        # 3. 采样
        next_tokens = self._sample(
            hidden_states,
            seq_group_metadata_list,
        )
        
        return next_tokens
    
    def _prepare_inputs(self, seq_group_metadata_list):
        """准备输入数据"""
        input_tokens = []
        input_positions = []
        slot_mapping = []
        
        for seq_group_metadata in seq_group_metadata_list:
            seq_data = seq_group_metadata.seq_data
            
            if seq_group_metadata.is_prompt:
                # Prefill阶段
                tokens = seq_data.get_token_ids()
                input_tokens.extend(tokens)
                input_positions.extend(range(len(tokens)))
            else:
                # Decode阶段
                tokens = [seq_data.get_last_token_id()]
                input_tokens.extend(tokens)
                input_positions.extend([seq_data.get_len() - 1])
            
            # 构造slot_mapping（KV Cache位置）
            block_table = seq_group_metadata.block_tables[seq_id]
            for i, token_id in enumerate(tokens):
                block_idx = i // self.block_size
                block_offset = i % self.block_size
                slot = block_table[block_idx] * self.block_size + block_offset
                slot_mapping.append(slot)
        
        return (
            torch.tensor(input_tokens),
            torch.tensor(input_positions),
            torch.tensor(slot_mapping),
        )
```

### 4.4 CUDA Graph 优化

**核心文件**：`vllm/worker/model_runner.py`

```python
class CUDAGraphRunner:
    """CUDA Graph优化的模型执行器"""
    
    def __init__(self, model):
        self.model = model
        self.graph_pool = {}  # {batch_size: CUDAGraph}
    
    def capture(self, batch_size: int):
        """捕获CUDA Graph"""
        # 1. 准备固定大小的输入
        input_tokens = torch.zeros((batch_size,), dtype=torch.long, device="cuda")
        input_positions = torch.zeros((batch_size,), dtype=torch.long, device="cuda")
        
        # 2. Warmup
        for _ in range(3):
            _ = self.model(input_tokens, input_positions)
        
        # 3. 捕获Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(input_tokens, input_positions)
        
        self.graph_pool[batch_size] = {
            "graph": graph,
            "input_tokens": input_tokens,
            "input_positions": input_positions,
            "output": output,
        }
    
    def forward(self, input_tokens, input_positions):
        """使用CUDA Graph执行"""
        batch_size = input_tokens.shape[0]
        
        # 如果没有对应的Graph，捕获一个
        if batch_size not in self.graph_pool:
            self.capture(batch_size)
        
        graph_data = self.graph_pool[batch_size]
        
        # 复制输入
        graph_data["input_tokens"].copy_(input_tokens)
        graph_data["input_positions"].copy_(input_positions)
        
        # 重放Graph
        graph_data["graph"].replay()
        
        return graph_data["output"]
```

---

## 5. 生产环境最佳实践

### 5.1 容量规划

**计算公式**：
```
GPU Memory = Model Weights + KV Cache + Activations + Overhead

Model Weights = num_params × bytes_per_param
KV Cache = max_num_seqs × max_model_len × kv_cache_per_token
Activations = batch_size × seq_len × hidden_size × bytes_per_activation
Overhead ≈ 2-4 GB
```

**示例**（Llama-2-7B, FP16, A100 80GB）：
```
Model Weights = 7B × 2 = 14 GB
KV Cache per token = 2 × 32 layers × 4096 hidden × 2 bytes = 0.5 MB
Max KV Cache = (80 - 14 - 4) × 0.95 = 59 GB
Max Concurrent Seqs = 59 GB / (2048 tokens × 0.5 MB) ≈ 57
```

### 5.2 参数调优

**高吞吐量场景**：
```bash
--max-num-seqs 256 \
--max-num-batched-tokens 8192 \
--gpu-memory-utilization 0.95
```

**低延迟场景**：
```bash
--max-num-seqs 32 \
--max-num-batched-tokens 2048 \
--gpu-memory-utilization 0.85
```

**Few-shot场景**：
```bash
--enable-prefix-caching \
--max-num-seqs 64 \
--block-size 16
```

### 5.3 监控指标

**关键指标**：
```python
# 吞吐量
tokens_per_second = total_tokens / total_time

# 延迟
ttft = time_to_first_token
tpot = time_per_output_token

# GPU利用率
gpu_utilization = gpu_active_time / total_time

# KV Cache利用率
kv_cache_utilization = used_blocks / total_blocks
```

**Prometheus监控**：
```python
from prometheus_client import Counter, Histogram

# 请求计数
request_counter = Counter('vllm_requests_total', 'Total requests')

# 延迟分布
ttft_histogram = Histogram('vllm_ttft_seconds', 'TTFT distribution')
tpot_histogram = Histogram('vllm_tpot_seconds', 'TPOT distribution')

# GPU利用率
gpu_utilization_gauge = Gauge('vllm_gpu_utilization', 'GPU utilization')
```

### 5.4 故障排查

**问题1：OOM (Out of Memory)**
```
症状：CUDA out of memory
原因：max_num_seqs或max_model_len过大

解决：
1. 降低 --gpu-memory-utilization (0.9 → 0.85)
2. 降低 --max-num-seqs
3. 降低 --max-model-len
4. 增加GPU数量（TP）
```

**问题2：低吞吐量**
```
症状：tokens/s远低于预期
原因：batch size过小，GPU未充分利用

解决：
1. 增加 --max-num-seqs
2. 增加 --max-num-batched-tokens
3. 启用 --enable-prefix-caching
```

**问题3：高延迟**
```
症状：TTFT或TPOT过高
原因：batch size过大，Prefill阻塞Decode

解决：
1. 降低 --max-num-batched-tokens
2. 使用Chunked Prefill
3. 考虑PD分离架构
```

---

## 总结

### 核心要点

1. **框架选择**：
   - 通用场景：vLLM
   - Few-shot/对话：SGLang
   - 极致性能：TensorRT-LLM

2. **vLLM核心设计**：
   - PagedAttention：块级KV Cache管理
   - Continuous Batching：动态批处理
   - CUDA Graph：减少launch overhead

3. **关键参数**：
   - `--gpu-memory-utilization`: 0.85-0.95
   - `--max-num-seqs`: 根据内存计算
   - `--enable-prefix-caching`: Few-shot场景必开

4. **生产实践**：
   - 容量规划：精确计算内存需求
   - 参数调优：根据场景选择
   - 监控告警：关注吞吐量和延迟

---

**下一部分**：量化技术基础
