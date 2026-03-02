# 大模型推理技术分享 - Part 2: GPU与算子相关

> **受众**: AI平台开发工程师
> **目标**: 理解大模型推理中的GPU架构和算子优化

---

## 目录

1. [GPU 架构基础](#1-gpu-架构基础)
2. [内存层次结构](#2-内存层次结构)
3. [CUDA 编程模型](#3-cuda-编程模型)
4. [关键算子优化](#4-关键算子优化)
5. [性能分析与调优](#5-性能分析与调优)

---

## 1. GPU 架构基础

### 1.1 GPU vs CPU

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数 | 8-64 | 数千-数万 |
| 时钟频率 | 3-5 GHz | 1-2 GHz |
| 设计目标 | 低延迟 | 高吞吐量 |
| 适用场景 | 串行计算 | 并行计算 |
| 内存带宽 | ~100 GB/s | ~2000 GB/s |

**为什么 GPU 适合大模型推理？**
- 矩阵乘法：高度并行
- 内存带宽：KV Cache 访问密集
- 吞吐量优先：批处理推理

### 1.2 NVIDIA GPU 架构演进

#### A100 (Ampere 架构)
```
规格：
- SM (Streaming Multiprocessor): 108 个
- CUDA Cores: 6912 个
- Tensor Cores: 432 个（第3代）
- HBM2e: 40GB/80GB
- 内存带宽: 1.6 TB/s (40GB) / 2.0 TB/s (80GB)
- FP16 性能: 312 TFLOPS
- INT8 性能: 624 TOPS
```

**关键特性**：
- **Multi-Instance GPU (MIG)**: 将 GPU 分割为多个独立实例
- **稀疏性加速**: Tensor Core 支持 2:4 结构化稀疏
- **TF32**: 自动加速 FP32 计算

#### H100 (Hopper 架构)
```
规格：
- SM: 132 个
- CUDA Cores: 16896 个
- Tensor Cores: 528 个（第4代）
- HBM3: 80GB
- 内存带宽: 3.35 TB/s
- FP16 性能: 989 TFLOPS
- FP8 性能: 1979 TFLOPS
```

**关键特性**：
- **Transformer Engine**: 原生支持 FP8
- **Thread Block Clusters**: 更好的 SM 间协作
- **DPX Instructions**: 动态编程加速

### 1.3 SM (Streaming Multiprocessor) 架构

**A100 SM 结构**：
```
SM (每个)
├─ 4 个 Processing Blocks
│   ├─ 16 个 FP32 CUDA Cores
│   ├─ 16 个 INT32 Cores
│   ├─ 8 个 FP64 Cores
│   └─ 1 个 Tensor Core
├─ 4 个 Warp Schedulers
├─ 192 KB L1 Cache / Shared Memory
└─ 64K 32-bit Registers
```

**关键概念**：
- **Warp**: 32 个线程的执行单元
- **Warp Scheduler**: 每个周期调度一个 warp
- **Occupancy**: SM 上活跃 warp 的比例

---

## 2. 内存层次结构

### 2.1 内存层次

```
┌─────────────────────────────────────────────────────┐
│  Registers (寄存器)                                  │
│  - 延迟: 1 cycle                                     │
│  - 容量: 64K × 32-bit per SM                        │
│  - 带宽: ~20 TB/s                                    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  Shared Memory / L1 Cache (共享内存/L1缓存)          │
│  - 延迟: ~20 cycles                                  │
│  - 容量: 192 KB per SM (A100)                       │
│  - 带宽: ~19 TB/s                                    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  L2 Cache (L2缓存)                                   │
│  - 延迟: ~200 cycles                                 │
│  - 容量: 40 MB (A100)                               │
│  - 带宽: ~7 TB/s                                     │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│  HBM (High Bandwidth Memory)                        │
│  - 延迟: ~300-400 cycles                             │
│  - 容量: 40/80 GB (A100)                            │
│  - 带宽: 1.6/2.0 TB/s                               │
└─────────────────────────────────────────────────────┘
```

### 2.2 内存带宽的重要性

**Decode 阶段的瓶颈**：
```python
# Decode: 生成 1 个 token
# 需要读取的数据：
# - 模型权重: ~7B × 2 bytes = 14 GB (FP16)
# - KV Cache: seq_len × hidden_size × num_layers × 2 × 2 bytes

# 计算量：
FLOPs = 2 × 7B ≈ 14 GFLOP

# 内存访问量：
Memory = 14 GB (权重) + KV Cache

# 计算强度 (Arithmetic Intensity):
AI = FLOPs / Memory ≈ 14 GFLOP / 14 GB = 1 FLOP/Byte
```

**Roofline 模型**：
```
性能 = min(Peak FLOPS, Memory Bandwidth × AI)

对于 Decode:
- Peak FLOPS: 312 TFLOPS (A100 FP16)
- Memory Bandwidth: 2 TB/s
- AI: 1 FLOP/Byte

实际性能 = min(312 TFLOPS, 2 TB/s × 1) = 2 TFLOPS
GPU 利用率 = 2 / 312 = 0.6%
```

**结论**：Decode 阶段是 **Memory-bound**，优化重点是减少内存访问。

### 2.3 内存访问模式

**Coalesced Access（合并访问）**：
```
Good (合并访问):
Thread 0: 读取 addr[0]
Thread 1: 读取 addr[1]
Thread 2: 读取 addr[2]
...
Thread 31: 读取 addr[31]
→ 1 次内存事务

Bad (非合并访问):
Thread 0: 读取 addr[0]
Thread 1: 读取 addr[100]
Thread 2: 读取 addr[200]
...
→ 32 次内存事务
```

**Bank Conflict（Bank 冲突）**：
```
Shared Memory 分为 32 个 Bank

Good (无冲突):
Thread 0: 读取 smem[0]  (Bank 0)
Thread 1: 读取 smem[1]  (Bank 1)
...
→ 并行访问

Bad (2-way 冲突):
Thread 0: 读取 smem[0]  (Bank 0)
Thread 1: 读取 smem[32] (Bank 0)
→ 串行访问，性能降低 2x
```

---

## 3. CUDA 编程模型

### 3.1 线程层次

```
Grid (网格)
├─ Block 0
│   ├─ Warp 0 (Thread 0-31)
│   ├─ Warp 1 (Thread 32-63)
│   └─ ...
├─ Block 1
│   └─ ...
└─ ...
```

**示例**：矩阵乘法
```cuda
// Kernel 启动
dim3 grid(M/16, N/16);   // Grid: (M/16) × (N/16) blocks
dim3 block(16, 16);      // Block: 16 × 16 threads
matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

// Kernel 实现
__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 3.2 Shared Memory 优化

**朴素实现**（每个线程从 Global Memory 读取）：
```cuda
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // 每次从 Global Memory 读取
    }
    C[row * N + col] = sum;
}
```

**优化实现**（使用 Shared Memory）：
```cuda
__global__ void matmul_shared(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // 1. 加载 tile 到 Shared Memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 2. 计算（从 Shared Memory 读取）
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

**性能提升**：
- 朴素实现：每个元素从 Global Memory 读取 N 次
- 优化实现：每个元素从 Global Memory 读取 1 次，从 Shared Memory 读取 N 次
- 加速比：~10x

### 3.3 Tensor Core

**什么是 Tensor Core？**
- 专用于矩阵乘法的硬件单元
- 一次计算 4×4 或 8×8 矩阵乘法
- 性能远超 CUDA Core

**性能对比**（A100）：
- CUDA Core (FP16): 19.5 TFLOPS
- Tensor Core (FP16): 312 TFLOPS
- **加速比**: 16x

**使用 Tensor Core**：
```cuda
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void matmul_wmma(half* A, half* B, float* C, int M, int N, int K) {
    // 声明 fragment（寄存器中的矩阵片段）
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    
    // 初始化累加器
    fill_fragment(c_frag, 0.0f);
    
    // 分块计算
    for (int k = 0; k < K; k += 16) {
        // 加载 A 和 B 的 fragment
        load_matrix_sync(a_frag, A + ..., K);
        load_matrix_sync(b_frag, B + ..., N);
        
        // 矩阵乘法（使用 Tensor Core）
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // 存储结果
    store_matrix_sync(C + ..., c_frag, N, mem_row_major);
}
```

---

## 4. 关键算子优化

### 4.1 MatMul（矩阵乘法）

**大模型中的 MatMul**：
```
Linear 层: Y = X @ W
- X: [batch_size, seq_len, hidden_size]
- W: [hidden_size, hidden_size]
- Y: [batch_size, seq_len, hidden_size]

FLOPs = 2 × batch_size × seq_len × hidden_size²
```

**优化策略**：

1. **使用 Tensor Core**：
   - 要求：FP16/BF16/FP8 数据类型
   - 对齐：矩阵维度是 8 的倍数
   - 库：cuBLAS, CUTLASS

2. **Tiling（分块）**：
   - 将大矩阵分块，提高 Cache 命中率
   - 块大小：通常 64×64 或 128×128

3. **Fusion（算子融合）**：
   ```python
   # 未融合
   x = matmul(input, weight)
   x = bias_add(x, bias)
   x = gelu(x)
   
   # 融合
   x = fused_linear_gelu(input, weight, bias)
   ```

**cuBLAS 示例**：
```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// 设置为使用 Tensor Core
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// 矩阵乘法: C = alpha * A @ B + beta * C
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    A, CUDA_R_16F, M,
    B, CUDA_R_16F, K,
    &beta,
    C, CUDA_R_16F, M,
    CUDA_R_32F,  // 计算类型
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

### 4.2 Softmax

**标准实现**：
```python
def softmax(x):
    # x: [batch, seq_len]
    exp_x = torch.exp(x - x.max(dim=-1, keepdim=True))
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

**问题**：
- 需要 3 次 pass：max, exp+sum, div
- 中间结果需要存储

**优化：Online Softmax**：
```cuda
__global__ void online_softmax(float* input, float* output, int N) {
    int row = blockIdx.x;
    float* x = input + row * N;
    float* y = output + row * N;
    
    // 1. 在线计算 max 和 sum
    float max_val = -INFINITY;
    float sum = 0.0f;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = x[i];
        
        // 更新 max
        float old_max = max_val;
        max_val = fmaxf(max_val, val);
        
        // 更新 sum（考虑 max 的变化）
        sum = sum * expf(old_max - max_val) + expf(val - max_val);
    }
    
    // 2. Warp reduce
    max_val = warp_reduce_max(max_val);
    sum = warp_reduce_sum(sum);
    
    // 3. 计算输出
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}
```

**优势**：
- 只需 1 次 pass
- 不需要存储中间结果
- FlashAttention 的核心技术

### 4.3 LayerNorm

**标准实现**：
```python
def layer_norm(x, weight, bias, eps=1e-5):
    # x: [batch, seq_len, hidden_size]
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return weight * x_norm + bias
```

**优化：Welford's Online Algorithm**：
```cuda
__global__ void layer_norm(float* input, float* output, float* weight, float* bias, int N) {
    int row = blockIdx.x;
    float* x = input + row * N;
    float* y = output + row * N;
    
    // 1. 在线计算 mean 和 variance (Welford's algorithm)
    float mean = 0.0f;
    float m2 = 0.0f;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = x[i];
        float delta = val - mean;
        mean += delta / (i + 1);
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // 2. Warp reduce
    mean = warp_reduce_sum(mean) / N;
    float var = warp_reduce_sum(m2) / N;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // 3. Normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x_norm = (x[i] - mean) * inv_std;
        y[i] = weight[i] * x_norm + bias[i];
    }
}
```

### 4.4 FlashAttention Kernel

**核心思想**：
- 分块计算，减少 HBM 访问
- 在线 Softmax，不存储 attention matrix

**伪代码**：
```cuda
__global__ void flash_attention(
    float* Q, float* K, float* V, float* O,
    int seq_len, int head_dim
) {
    // 每个 block 处理一个 query block
    __shared__ float Q_shared[BLOCK_SIZE][HEAD_DIM];
    __shared__ float K_shared[BLOCK_SIZE][HEAD_DIM];
    __shared__ float V_shared[BLOCK_SIZE][HEAD_DIM];
    
    // 加载 Q block 到 shared memory
    load_Q_block(Q_shared, Q, ...);
    
    // 初始化输出和 softmax 统计量
    float O_local[HEAD_DIM] = {0};
    float max_val = -INFINITY;
    float sum = 0.0f;
    
    // 遍历所有 K, V blocks
    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        // 1. 加载 K, V block
        load_KV_block(K_shared, V_shared, K, V, k_block, ...);
        __syncthreads();
        
        // 2. 计算 Q @ K^T
        float scores[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            scores[i] = dot(Q_shared[threadIdx.x], K_shared[i]);
        }
        
        // 3. 在线更新 softmax 和输出
        float old_max = max_val;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            max_val = fmaxf(max_val, scores[i]);
        }
        
        // 重新缩放之前的输出
        float scale = expf(old_max - max_val);
        for (int i = 0; i < HEAD_DIM; i++) {
            O_local[i] *= scale;
        }
        sum *= scale;
        
        // 累加当前 block 的贡献
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float attn_weight = expf(scores[i] - max_val);
            sum += attn_weight;
            for (int j = 0; j < HEAD_DIM; j++) {
                O_local[j] += attn_weight * V_shared[i][j];
            }
        }
        __syncthreads();
    }
    
    // 4. 最终归一化
    for (int i = 0; i < HEAD_DIM; i++) {
        O_local[i] /= sum;
    }
    
    // 5. 写回 global memory
    store_output(O, O_local, ...);
}
```

**性能分析**：
```
标准 Attention:
- HBM 访问: O(seq_len²) + O(seq_len × head_dim)
- 时间: ~100 ms (seq_len=2048)

FlashAttention:
- HBM 访问: O(seq_len² / BLOCK_SIZE) + O(seq_len × head_dim)
- 时间: ~30 ms (seq_len=2048)
- 加速比: 3.3x
```

---

## 5. 性能分析与调优

### 5.1 性能指标

**吞吐量指标**：
- **TFLOPS**: 每秒万亿次浮点运算
- **Tokens/s**: 每秒生成的 token 数
- **Requests/s**: 每秒处理的请求数

**延迟指标**：
- **TTFT (Time To First Token)**: 首 token 延迟
- **TPOT (Time Per Output Token)**: 每个输出 token 的延迟
- **E2E Latency**: 端到端延迟

**效率指标**：
- **GPU Utilization**: GPU 利用率
- **Memory Bandwidth Utilization**: 内存带宽利用率
- **MFU (Model FLOPs Utilization)**: 模型 FLOPs 利用率

### 5.2 Profiling 工具

#### NVIDIA Nsight Systems
```bash
# 采集 profile
nsys profile -o profile.qdrep python inference.py

# 查看 timeline
nsys-ui profile.qdrep
```

**关键指标**：
- Kernel 执行时间
- 内存拷贝时间
- CPU-GPU 同步开销

#### NVIDIA Nsight Compute
```bash
# 采集单个 kernel 的详细信息
ncu --set full -o kernel_profile python inference.py

# 查看报告
ncu-ui kernel_profile.ncu-rep
```

**关键指标**：
- SM Efficiency
- Memory Throughput
- Warp Occupancy
- Register/Shared Memory Usage

#### PyTorch Profiler
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model(input)

# 打印统计信息
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 导出 Chrome trace
prof.export_chrome_trace("trace.json")
```

### 5.3 常见性能问题

#### 问题 1: Kernel Launch Overhead

**现象**：
- 大量小 kernel，每个执行时间很短
- CPU-GPU 同步频繁

**解决方案**：
1. **Kernel Fusion**: 合并多个 kernel
2. **CUDA Graph**: 减少 launch overhead
   ```python
   # 捕获 CUDA Graph
   g = torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):
       output = model(input)
   
   # 重放 Graph
   g.replay()
   ```

#### 问题 2: Memory Bandwidth Bottleneck

**现象**：
- GPU 利用率低（<20%）
- Memory Throughput 接近峰值

**解决方案**：
1. **算子融合**: 减少中间结果的读写
2. **量化**: 使用 INT8/FP8，减少内存访问
3. **KV Cache 优化**: PagedAttention, RadixAttention

#### 问题 3: Low Occupancy

**现象**：
- Warp Occupancy < 50%
- Register/Shared Memory 使用过多

**解决方案**：
1. **减少 Register 使用**: 简化 kernel 逻辑
2. **调整 Block Size**: 增加每个 SM 的 block 数
3. **减少 Shared Memory**: 使用更小的 tile size

### 5.4 优化 Checklist

**数据类型**：
- [ ] 使用 FP16/BF16 而非 FP32
- [ ] 考虑 FP8（H100）
- [ ] 考虑 INT8 量化

**算子优化**：
- [ ] 使用 Tensor Core（cuBLAS, CUTLASS）
- [ ] 算子融合（Fused Linear, Fused Attention）
- [ ] 使用优化库（FlashAttention, xFormers）

**内存优化**：
- [ ] KV Cache 管理（PagedAttention, RadixAttention）
- [ ] 减少内存拷贝（Pin Memory, Zero-Copy）
- [ ] 内存池（避免频繁分配/释放）

**并行优化**：
- [ ] Tensor Parallelism（单节点）
- [ ] Pipeline Parallelism（跨节点）
- [ ] Continuous Batching（动态批处理）

**编译优化**：
- [ ] torch.compile（PyTorch 2.0+）
- [ ] CUDA Graph（减少 launch overhead）
- [ ] TensorRT（静态图优化）

---

## 总结

### 核心要点

1. **GPU 架构**：
   - 高并行度（数千核心）
   - 高内存带宽（2+ TB/s）
   - Tensor Core（16x 加速）

2. **内存层次**：
   - Registers > Shared Memory > L2 Cache > HBM
   - Decode 阶段是 Memory-bound
   - 优化重点：减少 HBM 访问

3. **CUDA 编程**：
   - 线程层次：Grid > Block > Warp > Thread
   - Shared Memory：减少 Global Memory 访问
   - Tensor Core：专用矩阵乘法硬件

4. **关键算子**：
   - MatMul：使用 Tensor Core + Tiling
   - Softmax：Online Softmax（1-pass）
   - LayerNorm：Welford's Algorithm
   - FlashAttention：分块 + 在线 Softmax

5. **性能调优**：
   - Profiling：Nsight Systems/Compute
   - 常见问题：Launch Overhead, Memory Bottleneck, Low Occupancy
   - 优化方向：数据类型、算子融合、内存管理、并行策略

---

**下一部分**：推理框架对比与 vLLM 深入解析
