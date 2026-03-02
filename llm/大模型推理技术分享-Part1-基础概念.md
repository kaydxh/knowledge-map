# 大模型推理技术分享 - Part 1: 关键基础概念

> **受众**: AI平台开发工程师
> **目标**: 深入理解大模型推理的核心技术原理

---

## 目录

1. [KV Cache：推理加速的核心](#1-kv-cache推理加速的核心)
2. [Prefill vs Decode：两阶段推理](#2-prefill-vs-decode两阶段推理)
3. [FlashAttention：内存高效的注意力机制](#3-flashattention内存高效的注意力机制)
4. [Continuous Batching：动态批处理](#4-continuous-batching动态批处理)
5. [并行策略：TP、PP、DP](#5-并行策略tpppdp)
6. [推测解码：加速生成](#6-推测解码加速生成)

---

## 1. KV Cache：推理加速的核心

### 1.1 为什么需要 KV Cache？

**问题背景**：
- Transformer 的自回归生成：每次生成一个 token，需要用到所有历史 token
- 朴素实现：每次都重新计算所有历史 token 的 Key 和 Value
- **时间复杂度**：生成 n 个 token 需要 O(n²) 次计算

**示例**：生成 "Hello world"
```
Step 1: 输入 "Hello"     → 计算 K₁, V₁ → 生成 " "
Step 2: 输入 "Hello "    → 重新计算 K₁, V₁, K₂, V₂ → 生成 "world"
Step 3: 输入 "Hello world" → 重新计算 K₁, V₁, K₂, V₂, K₃, V₃ → ...
```

### 1.2 KV Cache 原理

**核心思想**：缓存已计算的 Key 和 Value，避免重复计算

```python
# 伪代码示例
class Attention:
    def __init__(self):
        self.kv_cache = {}  # 缓存 {position: (K, V)}
    
    def forward(self, x, position):
        # 计算当前 token 的 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 保存到 cache
        self.kv_cache[position] = (K, V)
        
        # 从 cache 获取所有历史 K, V
        all_K = [self.kv_cache[i][0] for i in range(position + 1)]
        all_V = [self.kv_cache[i][1] for i in range(position + 1)]
        
        # 计算 attention
        scores = Q @ concat(all_K).T
        attn = softmax(scores) @ concat(all_V)
        return attn
```

**优化效果**：
- 时间复杂度：O(n²) → O(n)
- 生成 100 个 token：从 5050 次计算 → 100 次计算

### 1.3 KV Cache 的内存开销

**计算公式**：
```
KV Cache 大小 = 2 × num_layers × seq_len × hidden_size × precision
```

**示例**：Llama-2-7B
- num_layers = 32
- hidden_size = 4096
- seq_len = 2048
- precision = FP16 (2 bytes)

```
KV Cache = 2 × 32 × 2048 × 4096 × 2 bytes
         = 1,073,741,824 bytes
         ≈ 1 GB per request
```

**挑战**：
- 单个请求就需要 1GB 显存
- A100 (80GB) 理论上只能并发 80 个请求
- 实际更少（模型权重、激活值也需要显存）

### 1.4 KV Cache 管理策略

#### 1.4.1 vLLM: PagedAttention

**核心思想**：借鉴操作系统的虚拟内存管理

```
逻辑地址空间（连续）          物理地址空间（分散）
┌─────────────────┐          ┌─────┐
│ Token 0-15      │ ───────> │Block│ (GPU Memory)
├─────────────────┤          ├─────┤
│ Token 16-31     │ ───────> │Block│
├─────────────────┤          ├─────┤
│ Token 32-47     │ ───────> │Block│
└─────────────────┘          └─────┘
```

**特点**：
- 固定块大小（如 16 tokens）
- 块表（Block Table）映射逻辑地址到物理地址
- 支持块共享（前缀共享）
- 减少内存碎片

**代码示例**（vLLM）：
```python
class BlockSpaceManager:
    def __init__(self, block_size=16, num_blocks=1000):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # {seq_id: [block_ids]}
    
    def allocate(self, seq_id, num_tokens):
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
        return blocks
    
    def free(self, seq_id):
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
```

#### 1.4.2 SGLang: RadixAttention

**核心思想**：使用 Radix Tree（前缀树）管理 KV Cache

```
示例：三个请求
- "Translate to French: Hello"
- "Translate to French: Goodbye"
- "Translate to Spanish: Hello"

Radix Tree:
Root
├─ "Translate to French: " (cached)
│   ├─ "Hello" (cached)
│   └─ "Goodbye" (cached)
└─ "Translate to Spanish: " (cached)
    └─ "Hello" (cached)
```

**特点**：
- 自动识别公共前缀
- 任意长度前缀（不限于固定块）
- 适合 Few-shot、多轮对话、Agent 场景
- LRU/LFU 驱逐策略

**代码示例**（SGLang）：
```python
class RadixCache:
    def __init__(self):
        self.root = TreeNode()
    
    def match_prefix(self, token_ids):
        """匹配最长公共前缀"""
        node = self.root
        matched_len = 0
        
        for i, token_id in enumerate(token_ids):
            if token_id in node.children:
                node = node.children[token_id]
                matched_len = i + 1
            else:
                break
        
        return matched_len, node
    
    def insert(self, token_ids, kv_data):
        """插入新的 KV Cache"""
        matched_len, node = self.match_prefix(token_ids)
        
        # 插入新节点
        for token_id in token_ids[matched_len:]:
            new_node = TreeNode(token_id)
            node.children[token_id] = new_node
            node = new_node
        
        node.kv_data = kv_data
```

### 1.5 对比总结

| 特性 | PagedAttention (vLLM) | RadixAttention (SGLang) |
|------|----------------------|------------------------|
| 数据结构 | Block Table | Radix Tree |
| 粒度 | 固定块（16 tokens） | 任意长度前缀 |
| 前缀共享 | 需要显式指定 | 自动识别 |
| 内存碎片 | 可能存在 | 更少 |
| 适用场景 | 通用推理 | Few-shot、对话、Agent |

---

## 2. Prefill vs Decode：两阶段推理

### 2.1 两阶段概述

**Prefill（预填充）**：
- 输入：完整的 prompt（如 100 个 tokens）
- 输出：第一个生成的 token
- 特点：**计算密集型**（Compute-bound）
- 并行度：高（所有 tokens 并行计算）

**Decode（解码）**：
- 输入：上一步生成的 token（1 个 token）
- 输出：下一个生成的 token
- 特点：**内存密集型**（Memory-bound）
- 并行度：低（只有 1 个 token）

### 2.2 计算特性对比

```
Prefill 阶段：
输入: [Token₁, Token₂, ..., Token₁₀₀]
     ↓ 并行计算 Attention
输出: Token₁₀₁

Decode 阶段：
输入: [Token₁₀₁]
     ↓ 串行生成
输出: Token₁₀₂
     ↓
输入: [Token₁₀₂]
     ↓
输出: Token₁₀₃
     ...
```

**性能指标**：
- Prefill: 关注 **吞吐量**（tokens/s）
- Decode: 关注 **延迟**（ms/token）

### 2.3 计算量分析

**Prefill**：
```
FLOPs = 2 × seq_len² × hidden_size × num_layers
```
- seq_len = 100: 大量矩阵乘法
- GPU 利用率高（>80%）

**Decode**：
```
FLOPs = 2 × seq_len × hidden_size × num_layers
```
- seq_len = 1: 矩阵-向量乘法
- GPU 利用率低（<20%）
- 瓶颈：**内存带宽**（从 HBM 读取 KV Cache）

### 2.4 Prefill-Decode 分离（PD Disaggregation）

**核心思想**：将 Prefill 和 Decode 分离到不同的 GPU 集群

```
┌─────────────────────┐         ┌─────────────────────┐
│  Prefill Cluster    │         │  Decode Cluster     │
│  (高算力 GPU)        │ ──KV──> │  (高内存 GPU)        │
│  H100, A100         │         │  A100, A10          │
│  - 处理长上下文      │         │  - 大批量并发        │
│  - GPU 利用率高      │         │  - 内存带宽优化      │
└─────────────────────┘         └─────────────────────┘
```

**优势**：
- Prefill: 使用高算力 GPU（H100），快速处理长上下文
- Decode: 使用高内存 GPU（A100），支持大批量并发
- 资源利用率提升 30-50%

**SGLang 实现**：
```python
# Prefill Server
class PrefillServer:
    def process_request(self, prompt):
        # 1. Prefill 计算
        kv_cache = self.model.prefill(prompt)
        
        # 2. 传输 KV Cache 到 Decode Server
        self.send_kv_cache(kv_cache, decode_server)
        
        return first_token

# Decode Server
class DecodeServer:
    def process_request(self, kv_cache, first_token):
        # 1. 接收 KV Cache
        self.load_kv_cache(kv_cache)
        
        # 2. Decode 生成
        tokens = [first_token]
        while not done:
            next_token = self.model.decode(tokens[-1])
            tokens.append(next_token)
        
        return tokens
```

### 2.5 混合批处理的挑战

**问题**：Prefill 和 Decode 混合在同一个 batch 中

```
Batch:
- Request 1: Prefill (100 tokens)
- Request 2: Decode (1 token)
- Request 3: Decode (1 token)
```

**挑战**：
- Prefill 需要大量计算，Decode 需要低延迟
- 如何平衡两者？

**解决方案**：
1. **Chunked Prefill**：将 Prefill 分块处理
   ```
   Prefill (100 tokens) → 分为 4 个 chunk (25 tokens each)
   ```
2. **优先级调度**：Decode 优先，Prefill 填充空闲
3. **分离集群**：PD Disaggregation

---

## 3. FlashAttention：内存高效的注意力机制

### 3.1 标准 Attention 的问题

**标准实现**：
```python
def standard_attention(Q, K, V):
    # Q, K, V: [batch, seq_len, hidden_dim]
    
    # 1. 计算 attention scores
    scores = Q @ K.T  # [batch, seq_len, seq_len]
    
    # 2. Softmax
    attn_weights = softmax(scores)  # [batch, seq_len, seq_len]
    
    # 3. 加权求和
    output = attn_weights @ V  # [batch, seq_len, hidden_dim]
    
    return output
```

**问题**：
- `scores` 和 `attn_weights` 需要存储：O(seq_len²) 内存
- seq_len = 2048: 需要 2048² × 4 bytes = 16 MB（单个 head）
- 32 个 heads: 512 MB
- **内存瓶颈**：限制了 batch size 和 seq_len

### 3.2 FlashAttention 原理

**核心思想**：
1. **分块计算**（Tiling）：将 Q, K, V 分块加载到 SRAM
2. **在线 Softmax**：不存储完整的 attention matrix
3. **重计算**：反向传播时重新计算，而不是存储

**算法流程**：
```
1. 将 Q 分为 [Q₁, Q₂, ..., Qₘ]
2. 将 K, V 分为 [K₁, V₁], [K₂, V₂], ..., [Kₙ, Vₙ]

3. For each Qᵢ:
     For each (Kⱼ, Vⱼ):
         a. 加载 Qᵢ, Kⱼ, Vⱼ 到 SRAM
         b. 计算 Sᵢⱼ = Qᵢ @ Kⱼ.T
         c. 在线更新 Softmax 和输出
         d. 丢弃 Sᵢⱼ（不存储）
```

**内存优化**：
- 标准 Attention: O(seq_len²) HBM 访问
- FlashAttention: O(seq_len²/block_size) HBM 访问
- **加速 2-4x**，内存减少 10-20x

### 3.3 FlashAttention-2 改进

**优化点**：
1. **更好的并行化**：减少非矩阵乘法操作
2. **Work Partitioning**：优化 warp 级别的调度
3. **减少共享内存读写**

**性能提升**：
- 相比 FlashAttention-1: 加速 1.5-2x
- 相比标准 Attention: 加速 3-8x

### 3.4 FlashAttention 在推理中的应用

**vLLM 集成**：
```python
from vllm.attention import PagedAttention

class PagedAttentionWithFlash:
    def forward(self, query, key_cache, value_cache, block_tables):
        # 1. 从 block_tables 获取 KV Cache 位置
        # 2. 调用 FlashAttention kernel
        output = flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_table=block_tables,
        )
        return output
```

**SGLang 集成**：
```python
from sglang.srt.layers.radix_attention import RadixAttention

class RadixAttentionWithFlash:
    def forward(self, q, k, v, forward_batch):
        # 1. 从 RadixCache 获取 KV Cache 位置
        # 2. 调用 FlashAttention kernel
        output = self.attn_backend.forward(
            q, k, v, self, forward_batch
        )
        return output
```

---

## 4. Continuous Batching：动态批处理

### 4.1 传统静态批处理的问题

**静态批处理**：
```
Batch 1: [Req1, Req2, Req3, Req4]
- 所有请求同时开始
- 所有请求同时结束（等待最长的请求）
- GPU 利用率低（短请求完成后空闲）
```

**示例**：
```
Req1: 生成 10 tokens  ████████████░░░░░░░░░░░░░░░░░░░░
Req2: 生成 50 tokens  ████████████████████████████████
Req3: 生成 20 tokens  ████████████████░░░░░░░░░░░░░░░░
Req4: 生成 30 tokens  ████████████████████████░░░░░░░░

GPU:                  ████████████████████████████████
                      ↑ 前 10 步满载
                      ↑ 后 40 步利用率下降
```

### 4.2 Continuous Batching 原理

**核心思想**：动态添加和移除请求

```
Step 1: [Req1, Req2, Req3, Req4]
Step 2: [Req1, Req2, Req3, Req4]
...
Step 10: [Req1 完成, Req2, Req3, Req4, Req5 新增]
Step 11: [Req2, Req3, Req4, Req5]
...
```

**优势**：
- GPU 利用率提升 30-50%
- 吞吐量提升 2-3x
- 延迟降低（新请求不需要等待）

### 4.3 实现细节

**vLLM 实现**：
```python
class Scheduler:
    def __init__(self):
        self.waiting_queue = []
        self.running_batch = []
    
    def schedule(self):
        # 1. 移除已完成的请求
        self.running_batch = [
            req for req in self.running_batch 
            if not req.is_finished()
        ]
        
        # 2. 从 waiting_queue 添加新请求
        while len(self.running_batch) < max_batch_size:
            if not self.waiting_queue:
                break
            
            req = self.waiting_queue.pop(0)
            self.running_batch.append(req)
        
        return self.running_batch
```

**SGLang 实现**（增加前缀感知）：
```python
class Scheduler:
    def schedule(self):
        # 1. 移除已完成的请求
        self.filter_finished_requests()
        
        # 2. 按前缀匹配度排序（LPF 策略）
        self.waiting_queue.sort(
            key=lambda req: self.tree_cache.match_prefix(req.input_ids),
            reverse=True
        )
        
        # 3. 添加新请求（优先高匹配度）
        while self.can_add_request():
            req = self.waiting_queue.pop(0)
            self.running_batch.append(req)
        
        return self.running_batch
```

### 4.4 调度策略

**FCFS (First-Come-First-Serve)**：
- 按到达顺序调度
- 公平性好，但效率不一定最优

**LPF (Longest-Prefix-First)**：
- 优先调度前缀匹配度高的请求
- 最大化 KV Cache 命中率
- **SGLang 默认策略**

**Priority-based**：
- 支持请求优先级
- 适合多租户场景

---

## 5. 并行策略：TP、PP、DP

### 5.1 为什么需要并行？

**问题**：
- Llama-2-70B: 140 GB（FP16）
- A100 (80GB): 单卡放不下
- 需要多卡并行

### 5.2 Tensor Parallelism (TP)

**核心思想**：将单个 Tensor 切分到多个 GPU

**示例**：Linear 层并行
```
原始: Y = X @ W
     X: [batch, hidden]
     W: [hidden, hidden]
     Y: [batch, hidden]

TP (2 GPUs):
GPU 0: Y₀ = X @ W₀  (W₀: [hidden, hidden/2])
GPU 1: Y₁ = X @ W₁  (W₁: [hidden, hidden/2])

最终: Y = concat([Y₀, Y₁])
```

**Attention 层并行**：
```
原始: 32 个 attention heads

TP (4 GPUs):
GPU 0: heads 0-7
GPU 1: heads 8-15
GPU 2: heads 16-23
GPU 3: heads 24-31
```

**通信开销**：
- All-Reduce: 每层需要同步
- 通信量：O(hidden_size)
- 适合：**高带宽互联**（NVLink, InfiniBand）

**代码示例**：
```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_size):
        self.tp_size = tp_size
        self.tp_rank = get_tp_rank()
        
        # 每个 GPU 只存储部分权重
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features // tp_size)
        )
    
    def forward(self, x):
        # 本地计算
        output = x @ self.weight
        
        # All-Gather（收集所有 GPU 的结果）
        output = all_gather(output, group=tp_group)
        
        return output
```

### 5.3 Pipeline Parallelism (PP)

**核心思想**：将模型按层切分到多个 GPU

```
GPU 0: Layers 0-7
GPU 1: Layers 8-15
GPU 2: Layers 16-23
GPU 3: Layers 24-31
```

**朴素 PP（气泡问题）**：
```
GPU 0: [F₁][F₂][F₃][F₄]░░░░░░░░░░░░
GPU 1: ░░░░[F₁][F₂][F₃][F₄]░░░░░░░░
GPU 2: ░░░░░░░░[F₁][F₂][F₃][F₄]░░░░
GPU 3: ░░░░░░░░░░░░[F₁][F₂][F₃][F₄]
       ↑ 气泡（GPU 空闲）
```

**GPipe（Micro-batching）**：
```
将 batch 切分为多个 micro-batch

GPU 0: [F₁][F₂][F₃][F₄][F₅][F₆][F₇][F₈]
GPU 1: ░░░░[F₁][F₂][F₃][F₄][F₅][F₆][F₇]
GPU 2: ░░░░░░░░[F₁][F₂][F₃][F₄][F₅][F₆]
GPU 3: ░░░░░░░░░░░░[F₁][F₂][F₃][F₄][F₅]
       ↑ 气泡减少
```

**通信开销**：
- 点对点通信（P2P）
- 通信量：O(hidden_size × micro_batch_size)
- 适合：**跨节点**（以太网也可）

### 5.4 Data Parallelism (DP)

**核心思想**：每个 GPU 有完整模型，处理不同数据

```
GPU 0: 处理 Batch 0
GPU 1: 处理 Batch 1
GPU 2: 处理 Batch 2
GPU 3: 处理 Batch 3
```

**推理中的 DP**：
- 不需要梯度同步（训练才需要）
- 完全独立，无通信开销
- 适合：**高吞吐量场景**

**代码示例**：
```python
class DataParallelInference:
    def __init__(self, model, num_gpus):
        self.models = [
            copy.deepcopy(model).to(f"cuda:{i}")
            for i in range(num_gpus)
        ]
    
    def forward(self, batches):
        # 并行处理多个 batch
        outputs = []
        for i, batch in enumerate(batches):
            gpu_id = i % len(self.models)
            output = self.models[gpu_id](batch)
            outputs.append(output)
        
        return outputs
```

### 5.5 混合并行

**3D 并行**：TP + PP + DP

```
示例：8 个 GPU，Llama-2-70B
- TP = 2（每个模型切分到 2 个 GPU）
- PP = 2（模型切分为 2 个 stage）
- DP = 2（2 个副本）

拓扑：
DP Rank 0:
  PP Stage 0: [GPU 0, GPU 1] (TP)
  PP Stage 1: [GPU 2, GPU 3] (TP)

DP Rank 1:
  PP Stage 0: [GPU 4, GPU 5] (TP)
  PP Stage 1: [GPU 6, GPU 7] (TP)
```

**选择建议**：
- 单节点（8 GPUs）：TP = 8
- 多节点（4 nodes × 8 GPUs）：TP = 8, PP = 4
- 超大模型（1T+ params）：TP = 8, PP = 16, DP = 4

---

## 6. 推测解码：加速生成

### 6.1 为什么需要推测解码？

**问题**：
- Decode 阶段是串行的（一次生成一个 token）
- GPU 利用率低（<20%）
- 延迟高（每个 token 需要一次前向传播）

**核心思想**：
- 用小模型（draft model）快速生成多个候选 token
- 用大模型（target model）并行验证
- 接受正确的 token，拒绝错误的 token

### 6.2 Speculative Decoding 算法

**流程**：
```
1. Draft Model 生成 k 个候选 token
   输入: "Hello"
   输出: [" world", ",", " there"]  (k=3)

2. Target Model 并行验证
   输入: ["Hello world", "Hello,", "Hello there"]
   输出: [✓, ✗, ✗]  (只有第一个正确)

3. 接受正确的 token，继续生成
   接受: " world"
   下一轮从 "Hello world" 开始
```

**加速比**：
- 理论：k 倍（如果所有候选都正确）
- 实际：1.5-3 倍（取决于 draft model 的准确率）

### 6.3 Draft Model 选择

**方案 1：小模型**
- 使用 Llama-2-7B 作为 draft，Llama-2-70B 作为 target
- 优点：准确率高
- 缺点：需要额外的模型

**方案 2：Medusa**
- 在 target model 上增加多个 head，预测多个未来 token
- 优点：不需要额外模型
- 缺点：准确率较低

**方案 3：EAGLE**
- 使用自回归的 draft head
- 优点：准确率高于 Medusa
- 缺点：实现复杂

### 6.4 代码示例

```python
class SpeculativeDecoding:
    def __init__(self, draft_model, target_model, k=4):
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = k
    
    def generate(self, prompt):
        tokens = tokenize(prompt)
        
        while not done:
            # 1. Draft model 生成 k 个候选
            draft_tokens = []
            for _ in range(self.k):
                next_token = self.draft_model.generate(tokens)
                draft_tokens.append(next_token)
                tokens.append(next_token)
            
            # 2. Target model 并行验证
            # 构造 k 个候选序列
            candidates = [
                tokens[:-self.k] + draft_tokens[:i+1]
                for i in range(self.k)
            ]
            
            # 并行前向传播
            logits = self.target_model.forward_batch(candidates)
            
            # 3. 验证并接受
            accepted = 0
            for i in range(self.k):
                predicted = torch.argmax(logits[i])
                if predicted == draft_tokens[i]:
                    accepted += 1
                else:
                    break
            
            # 4. 更新 tokens
            tokens = tokens[:-self.k] + draft_tokens[:accepted]
        
        return tokens
```

### 6.5 实际效果

**Llama-2-70B + Llama-2-7B (draft)**：
- 加速比：2.1x
- Draft 准确率：60-70%

**Medusa (Llama-2-70B)**：
- 加速比：1.5x
- Draft 准确率：40-50%

---

## 总结

### 核心概念回顾

1. **KV Cache**：缓存历史 Key 和 Value，避免重复计算
   - PagedAttention (vLLM)：块级管理
   - RadixAttention (SGLang)：前缀树管理

2. **Prefill vs Decode**：
   - Prefill：计算密集型，高并行度
   - Decode：内存密集型，低并行度
   - PD 分离：提升资源利用率

3. **FlashAttention**：内存高效的 Attention
   - 分块计算，减少 HBM 访问
   - 加速 2-4x，内存减少 10-20x

4. **Continuous Batching**：动态批处理
   - 动态添加/移除请求
   - GPU 利用率提升 30-50%

5. **并行策略**：
   - TP：Tensor 切分，适合单节点
   - PP：层切分，适合跨节点
   - DP：数据并行，适合高吞吐量

6. **推测解码**：
   - 小模型生成，大模型验证
   - 加速 1.5-3x

---

**下一部分**：GPU 与算子相关知识
