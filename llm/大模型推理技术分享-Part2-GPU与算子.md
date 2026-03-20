# 大模型推理技术分享 - Part 2: GPU与算子相关

> **受众**: AI平台开发工程师
> **目标**: 理解大模型推理中的GPU架构和算子优化，让你在面对推理性能问题时能找到硬件层面的根因

---

## 目录

- [为什么你需要了解这些？](#为什么你需要了解这些)
- [概念关系图](#概念关系图)
1. [GPU 架构基础](#1-gpu-架构基础)
2. [内存层次结构](#2-内存层次结构)
3. [CUDA 编程模型](#3-cuda-编程模型)
4. [关键算子优化](#4-关键算子优化)
5. [性能分析与调优](#5-性能分析与调优)
- [🎯 速查卡片](#-速查卡片)

---

## 为什么你需要了解这些？

在 Part1 中，我们学了推理的核心概念（KV Cache、Prefill/Decode、FlashAttention 等）。但当你真正去优化推理性能时，会遇到这些更底层的问题：

| 你遇到的问题 | 你需要理解的概念 | 才知道… |
|-------------|----------------|--------|
| nsys 报告里全是 kernel 名，看不懂 | **SM 架构**（§1） | GPU 内部到底是怎么组织和执行的 |
| 模型用了 FP16 但 Tensor Core 利用率为 0 | **Tensor Core**（§1.3） | 怎样的矩阵运算才能触发 Tensor Core |
| Decode 阶段 GPU 利用率 <1%，算力浪费 | **内存层次 & Roofline**（§2） | 性能瓶颈不在算力，而在显存带宽 |
| 自定义 kernel 性能很差 | **CUDA 编程模型**（§3） | Shared Memory、合并访问等基础概念 |
| 想理解 FlashAttention 为什么快 | **关键算子优化**（§4） | MatMul/Softmax/LayerNorm 的 GPU 优化原理 |
| 不知道从哪下手做性能优化 | **Profiling 工具**（§5） | 用 Nsight 等工具定位真正的瓶颈 |

本文就是帮你补齐这些 GPU 和算子层面的"底层直觉"。

---

## 概念关系图

下图展示了本文所有核心概念之间的依赖关系。**向下的箭头表示"理解下面需要先理解上面"**：

```
                    ┌───────────────────────┐
                    │  §1 GPU 架构基础       │ ← 硬件基础
                    │  SM / Tensor Core      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ↓                 ↓                 ↓
     ┌────────────────┐ ┌───────────────┐ ┌──────────────────┐
     │ §2 内存层次    │ │ §3 CUDA       │ │ §4 关键算子优化   │
     │ HBM/SRAM/     │ │    编程模型    │ │ MatMul/Softmax   │
     │ Roofline      │ │ Thread/Block  │ │ LayerNorm/FA     │
     └───────┬────────┘ └───────┬───────┘ └────────┬─────────┘
             │                  │                   │
             └──────────────────┼───────────────────┘
                                ↓
                    ┌───────────────────────┐
                    │  §5 性能分析与调优     │ ← 综合应用
                    │  Profiling / 优化策略  │
                    └───────────────────────┘
```

💡 **推荐阅读顺序**：§1 → §2 → §3 → §4 → §5（按章节顺序即可）

如果你赶时间，只看 **§1（GPU 架构基础）+ §2（内存层次）** 就能理解 80% 的推理性能问题的根因。

---

## 1. GPU 架构基础

> 🟢 **一句话版本**：GPU 就是"人海战术"的芯片——CPU 像一个博士（什么都会但只有几个人），GPU 像几千个小学生（只会简单加减法但人多力量大）。大模型推理本质就是海量矩阵运算，正好适合 GPU 的"人海战术"。

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
  > Occupancy 反映了 SM 上"有多少个 warp 在排队等执行"，足够高时能有效隐藏内存延迟，但过度追求 100% 可能适得其反。

> 💼 **实战关联**：
> - 当你用 `nvidia-smi` 看到 A100 和 H100 时，核心区别是：H100 的 Tensor Core 支持 FP8（A100 不支持），且 HBM 带宽从 2 TB/s 提升到 3.35 TB/s——这直接决定了 Decode 速度上限
> - `ncu` 报告里的 `sm__throughput.avg.pct_of_peak_sustained_elapsed` 就是 SM 利用率——如果它很低但 Memory 利用率很高，说明是 Memory-bound
> - 理解 SM 架构后你就知道：为什么 `--tensor-parallel-size=2` 时性能不一定是单卡的 2 倍——因为 TP 需要跨 SM 通信（AllReduce），有额外开销

---

## 2. 内存层次结构

> 🟢 **一句话版本**：GPU 内部有从快到慢、从小到大的多级存储（寄存器 → SRAM → L2 → HBM），就像你身边的便签纸（小但秒取）→ 抽屉里的笔记本（稍大但要开抽屉）→ 楼下仓库的档案柜（很大但要走一趟）。推理优化的核心就是让数据尽量待在"便签纸"（SRAM）上，少去"楼下仓库"（HBM）。

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

> **关于 cycle（时钟周期）**：表中的 **cycle** 指 GPU 核心时钟跳动一次的时间，即 `1 cycle = 1 / GPU 核心频率`。例如 A100 Boost 频率约 1.41 GHz，则 1 cycle ≈ 0.71 ns。寄存器延迟仅 1 cycle，意味着数据在下一个时钟周期即可用，几乎零等待；而 HBM 访问需要 300-400 cycles，是寄存器的数百倍——这正是优化时要尽量把热数据放在寄存器和 Shared Memory 中的核心原因。

### 2.2 HBM 详解

**HBM（High Bandwidth Memory，高带宽内存）** 是一种专为高性能计算设计的 DRAM 内存技术，是现代 AI 加速器（如 NVIDIA GPU）中显存的主体。

#### 为什么需要 HBM？

传统的 GDDR 内存通过 PCB 走线与 GPU 芯片连接，引脚数量和频率受限，带宽增长遇到瓶颈。而大模型推理（尤其是 Decode 阶段）对内存带宽极度敏感——每生成一个 token 都需要从显存中读取大量的模型权重和 KV Cache 数据，带宽不够意味着 GPU 算力"喂不饱"。

> **GDDR（Graphics Double Data Rate）** 是专为显卡设计的高速 SDRAM，采用传统 2D 平面封装，多颗 GDDR 芯片通过 **PCB 走线（数厘米）** 与 GPU 互联，单芯片位宽仅 **32-bit**，只能靠提高时钟频率换取带宽。GDDR 各代演进：GDDR5（~8 GB/s/芯片）→ GDDR6（~16 GB/s）→ GDDR6X（~21 GB/s，PAM4 编码）→ GDDR7（~36 GB/s）。其带宽瓶颈来自三方面：**引脚数量有限**（总位宽难超 384-bit）、**信号完整性随走线长度恶化**（高频串扰和衰减）、**I/O 功耗随频率超线性增长**。但 GDDR 成本仅为 HBM 的 1/3~1/5，容量扩展灵活（加芯片即可），因此消费级 GPU（如 RTX 4090 使用 GDDR6X，总带宽 ~1 TB/s）仍广泛采用。**一句话总结：GDDR 是通过 PCB 走线连接 GPU 的传统平面封装显存，位宽窄、靠高频换带宽，成本低但带宽增长已接近物理极限；大模型推理对带宽的极端需求，推动了数据中心 GPU 从 GDDR 转向 HBM 架构。**

#### HBM 核心架构

HBM 采用 **3D 堆叠 + 硅中介层（Silicon Interposer）** 的封装方式：

> **GPU Die = GPU 的硅裸片**，是从晶圆（Wafer）上切下来的、集成了全部计算电路的那块硅芯片本体，封装后就是我们看到的 GPU 芯片。

```
         ┌──────────────┐
         │   GPU Die    │  ← GPU 计算芯片
         └──────┬───────┘
                │  (硅中介层 Silicon Interposer，超短互联)
    ┌───────────┼───────────┐
    │           │           │
┌───┴───┐  ┌───┴───┐  ┌───┴───┐
│ DRAM  │  │ DRAM  │  │ DRAM  │  ← 多个 DRAM Die 垂直堆叠
│ DRAM  │  │ DRAM  │  │ DRAM  │
│ DRAM  │  │ DRAM  │  │ DRAM  │
│ DRAM  │  │ DRAM  │  │ DRAM  │
│ Logic │  │ Logic │  │ Logic │  ← 底层逻辑 Die（控制器）
└───────┘  └───────┘  └───────┘
  Stack 1    Stack 2    Stack 3   ← 多个 HBM Stack
```

> **简单来说**：**DRAM** = 存储数据的最基本芯片单元（一层一片）；**Stack** = 多层 DRAM 堆起来的组合体（HBM 的基本模块）；**HBM 总显存** = 多个 Stack 并联工作。

**关键技术点**：

| 特性 | 说明 |
|------|------|
| **3D 堆叠** | 多层 DRAM Die 通过 TSV（硅通孔，Through-Silicon Via）垂直堆叠，通常 4-12 层 |
| **硅中介层** | GPU 和 HBM Stack 封装在同一块硅中介层上，互联距离极短（毫米级），延迟低、功耗小 |
| **超宽总线** | 每个 Stack 有 1024-bit 位宽的数据总线（对比 GDDR6 只有 32-bit/通道） |
| **多通道并行** | 多个 HBM Stack 并行工作，总带宽叠加 |

#### HBM 各代演进

| 世代 | 单 Stack 带宽 | 单 Stack 容量 | 堆叠层数 | 代表 GPU |
|------|-------------|-------------|---------|---------|
| **HBM** | 128 GB/s | 1 GB | 4 层 | — |
| **HBM2** | 256 GB/s | 8 GB | 8 层 | V100 (4×HBM2 = 900 GB/s) |
| **HBM2e** | 410 GB/s | 16 GB | 8 层 | A100 (5×HBM2e ≈ 2 TB/s) |
| **HBM3** | 600+ GB/s | 16 GB | 8-12 层 | H100 (5×HBM3 ≈ 3.35 TB/s) |
| **HBM3e** | 1+ TB/s | 36 GB | 12 层 | H200 (6×HBM3e ≈ 4.8 TB/s) |

> **HBM2e 要点**：HBM2e 中的 "e" 代表 **enhanced（增强版）**，是 HBM2 的改进版本，也是 A100 GPU 所使用的显存技术。相比 HBM2，HBM2e 通过更高的信号速率和优化的 I/O 设计，在保持 8 层堆叠不变的情况下，将单 Stack 带宽从 256 GB/s 提升至 410 GB/s，容量从 8 GB 翻倍至 16 GB。A100 配备 5 个 HBM2e Stack，其中 40GB 版本总带宽约 1.6 TB/s，80GB 版本总带宽约 2.0 TB/s。HBM2e 是 A100 时代平衡性能与成本的显存方案，其带宽在 Decode 阶段直接决定了推理生成速度的上限。

#### HBM 在 GPU 内存层次中的位置

HBM 是 GPU 内存体系中的 **片外大容量存储**，参见上方 [2.1 内存层次](#21-内存层次) 的层次图：

```
速度快 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → 容量大

Registers (~KB, ~1 cycle)
  → SRAM / Shared Memory (~192KB/SM, ~20 cycles)  ← FlashAttention 分块数据加载到这里
    → L2 Cache (~40 MB, ~200 cycles)
      → HBM (~40-80 GB, ~300-400 cycles)  ★ 模型权重、KV Cache、激活值都存在这里
```

#### HBM 与大模型推理的关系

在 Decode 阶段，每生成一个 token：
- 需要读取 **全部模型权重**（如 70B 模型 ≈ 140 GB）
- 需要读取 **该请求的全部 KV Cache**（可能 1 GB+）
- 但实际计算量很小（只是一个 token 的矩阵-向量乘法）

```
Decode 阶段的算术强度（Arithmetic Intensity）:

   计算量       2 × 模型参数量           很小的数
  ─────── = ────────────────────── = ──────────── → 极低
  访存量     模型参数量 + KV Cache     很大的数
```

**以 LLaMA-2 70B（FP16）为例，理解为什么分子"很小"、分母"很大"：**

| | 含义 | 数值 | H100 上耗时 |
|---|---|---|---|
| **分子（计算量）** | Decode 每步只处理 1 个 token，是矩阵×向量运算，FLOPs = `2 × 参数量` | 140 GFLOP | ~0.14 ms |
| **分母（访存量）** | 虽然只算 1 个 token，但必须把**整个模型权重**从 HBM 读一遍 | 140+ GB（70B×2bytes + KV Cache） | ~41.8 ms |

- **算术强度** ≈ 140 GFLOP / 140 GB ≈ **1 FLOP/Byte**
- 而 H100 的计算-带宽比为 990 TFLOPS / 3.35 TB/s ≈ **295 FLOP/Byte**
- GPU 有能力每读 1 字节做 295 次运算，但 Decode 只需做 1 次——**算力浪费了 99.7%，GPU 99% 的时间都在等内存搬数据**

这意味着 **GPU 的计算单元在等待 HBM 传输数据**，算力利用率很低。此时性能完全取决于 HBM 的带宽：

| GPU | HBM 带宽 | Decode 理论极限 (7B, FP16) |
|-----|---------|--------------------------|
| A100 | 2 TB/s | ~143 tokens/s |
| H100 | 3.35 TB/s | ~239 tokens/s |
| H200 | 4.8 TB/s | ~343 tokens/s |

> 计算方式：`HBM带宽 / (模型大小 × 2 bytes)` ≈ 每秒能完成的 Decode 步数

这就是为什么以下优化技术都在直接或间接地解决 HBM 带宽瓶颈：
1. **FlashAttention**：分块计算减少 HBM 访问次数
2. **量化**（INT8/INT4）：缩小模型和 KV Cache 体积，减轻带宽压力
3. **推测解码**：一次读取验证多个 token，提高每次 HBM 访问的产出
4. **GQA/MQA**：减少 KV Head 数来缩小 KV Cache，减少 HBM 读取量

#### 总结

**HBM 是 GPU 的"主内存"**，它决定了：
- **容量**：能装多大的模型、能并发多少请求（KV Cache 容量）
- **带宽**：Decode 阶段的生成速度上限

在大模型推理中，**HBM 带宽是 Decode 阶段最核心的性能瓶颈**。

### 2.3 内存带宽的重要性

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

### 2.4 内存访问模式

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

#### Coalesced Access 详解

GPU 中线程以 **Warp（32 个线程）** 为单位同时执行。当一个 Warp 发起内存读取时，内存控制器会尝试把 32 个请求**合并**成尽可能少的**内存事务（Memory Transaction）**——一次事务通常读取一个连续的 **128 字节**缓存行。

- **地址连续**：32 个线程各读 4 字节，恰好 128B → **1 次事务**，带宽利用率 100%
- **地址分散**（步长 = 100）：每个线程落在不同缓存行 → **最多 32 次事务**，带宽利用率 ~3%

> **实际场景**：行优先矩阵中，相邻线程读同一行的相邻列 → 合并访问 ✅；读同一列的相邻行 → 地址间隔 N，非合并访问 ❌。这也是 GPU 上矩阵转置是经典优化问题的原因。

#### Bank Conflict 详解

Shared Memory 被硬件分为 **32 个 Bank**（存储体），映射规则为 `Bank 编号 = (地址 / 4字节) % 32`。每个 Bank 每周期只能服务**一个地址**，因此：

- **无冲突**：32 个线程各访问不同 Bank → 1 个周期完成
- **N-way 冲突**：N 个线程访问同一 Bank 的不同地址 → 串行化为 N 个周期（性能降低 Nx）
- **广播例外**：多个线程访问同一 Bank 的**同一地址** → 硬件广播，不算冲突

> **常见解决方案——Padding**：分配 Shared Memory 时每行多加 1 个元素（如 `float tile[32][32+1]`），使列方向步长从 32 变为 33，错开 Bank 映射，消除列访问时的 Bank 冲突。

#### 两者的关系总结

| | Coalesced Access | Bank Conflict |
|---|---|---|
| **针对的内存** | 全局内存（HBM / Global Memory） | 共享内存（Shared Memory） |
| **问题本质** | 访问地址不连续，无法合并为少量事务 | 多个线程同时访问同一 Bank 的不同地址 |
| **后果** | 内存事务暴增（最多 32 倍） | 访问被串行化（最多 32 倍延迟） |
| **优化方向** | 保证 Warp 内相邻线程访问连续地址 | Padding / Swizzle 错开 Bank |

> **一句话总结**：合并访问解决"怎样高效地从 HBM 搬数据到芯片上"的问题，Bank 冲突解决"数据到了片上 Shared Memory 后，怎样让 32 个线程高效并行使用"的问题。两者分别卡住了 GPU 内存访问的**外部带宽**和**内部并行度**，都是算子优化的核心关注点。

> 💼 **实战关联**：
> - 当你看到 nsys 报告中某个 kernel 的 Memory Throughput 接近峰值（如 A100 的 2 TB/s）但 SM Utilization 很低时，这就是典型的 Memory-bound
> - vLLM 的 `--dtype float16` 参数直接决定了模型权重的大小，从而决定 Decode 阶段每步需要从 HBM 读取多少数据
> - 量化（INT8/INT4）的核心价值不是省显存（虽然也省），而是**减少 HBM 读取量**——模型体积缩小一半，Decode 速度理论上提升一倍
> - 理解内存层次后你就明白了：FlashAttention 之所以快，是因为它把 Attention 计算从 HBM（慢）搬到了 SRAM（快）

---

## 3. CUDA 编程模型

> 🟢 **一句话版本**：CUDA 是 NVIDIA 提供的 GPU 编程框架。你写一个函数（Kernel），GPU 会用几千个线程同时执行它。线程按 Thread → Warp（32个一组）→ Block → Grid 的层次组织，就像公司里的员工 → 小组 → 部门 → 公司。理解这个层次，才能写出高效的 GPU 代码。

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

> 💼 **实战关联**：
> - 当你用 `ncu` 分析某个 kernel 发现 Occupancy 很低时，通常是因为 Register 或 Shared Memory 用得太多，限制了每个 SM 能跑的 Block 数
> - 理解 Shared Memory 后你就能看懂 FlashAttention 的源码了——它本质就是把 Q/K/V 分块加载到 Shared Memory 中计算
> - Tensor Core 需要特定数据类型（FP16/BF16/FP8）和对齐要求（维度是 8 的倍数），如果你的模型 hidden_size 不满足这些条件，Tensor Core 不会被触发

---

## 4. 关键算子优化

> 🟢 **一句话版本**：大模型推理 90% 以上的时间花在 MatMul（矩阵乘法）、Softmax、LayerNorm 和 Attention 这几个核心算子上。优化它们的共同思路就两个字：**少搬**（减少显存读写）和**多算**（用 Tensor Core 加速计算）。

### 4.1 MatMul（矩阵乘法）

**大模型中的 MatMul**：
```
Linear 层: Y = X @ W
- X: [batch_size, seq_len, hidden_size]
- W: [hidden_size, hidden_size]
- Y: [batch_size, seq_len, hidden_size]

FLOPs = 2 × batch_size × seq_len × hidden_size²
```

**公式详解**：

各参数含义：

| 符号 | 含义 | 说明 |
|---|---|---|
| **batch_size** | 批大小 | 一次送入多少条样本 |
| **seq_len** | 序列长度 | 每条样本有多少个 token |
| **hidden_size** | 隐藏层维度 | 模型的宽度（如 LLaMA-7B 为 4096，70B 为 8192） |
| **hidden_size²** | 权重矩阵的大小 | W 是 `[hidden_size, hidden_size]`，参数量 = hidden_size² |

为什么系数是 **2**？

矩阵乘法 `Y = X @ W` 中，输出矩阵每个元素的计算是一个长度为 hidden_size 的点积：
```
Y[i][j] = X[i][0]×W[0][j] + X[i][1]×W[1][j] + ... + X[i][hidden_size-1]×W[hidden_size-1][j]
```
- hidden_size 次乘法 + hidden_size 次加法 = **2 × hidden_size 次浮点运算**

完整推导：
```
输出元素总数 = batch_size × seq_len × hidden_size
每个元素计算量 = 2 × hidden_size

总 FLOPs = (batch_size × seq_len × hidden_size) × (2 × hidden_size)
         = 2 × batch_size × seq_len × hidden_size²
```

实际例子（以 **LLaMA-2 7B** 的一个 Linear 层为例，hidden_size = 4096）：

| 场景 | batch_size | seq_len | FLOPs | 结果 |
|---|---|---|---|---|
| Decode 阶段（逐 token） | 1 | 1 | 2×1×1×4096² | **33.5 MFLOPs** |
| Prefill 阶段（2048 token） | 1 | 2048 | 2×1×2048×4096² | **68.7 GFLOPs** |

> **关键洞察**：Decode 时 seq_len=1，计算量极小（33.5M），但仍要读取整个权重矩阵（4096²×2 bytes = 32MB），这正是 **Memory-bound（访存瓶颈）** 的根源；Prefill 时 seq_len 很大，计算量放大 2048 倍，变为 **Compute-bound（计算瓶颈）**。

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

   **算子融合详解**：

   **未融合时的问题**：每个算子是一次独立的 GPU Kernel 调用，执行流程为 `HBM → 片上缓存 → 计算 → 片上缓存 → HBM`。3 个独立算子意味着 3 次显存读写往返：
   - **算子1 matmul**：从 HBM 读 input、weight → 计算 → 结果 x 写回 HBM
   - **算子2 bias_add**：从 HBM 读 x、bias → 计算 → 结果 x 写回 HBM
   - **算子3 gelu**：从 HBM 读 x → 计算 → 结果 x 写回 HBM

   `bias_add` 和 `gelu` 都是逐元素操作（Element-wise），计算量极小，但每次都要从 HBM 读写完整张量，是典型的 Memory-bound 操作。以 x 形状 `[1, 2048, 4096]`、FP16 为例：

   | 算子 | 计算量 | 显存读写量 | 算术强度 |
   |---|---|---|---|
   | matmul | ~68.7 GFLOPs | ~48 MB | **高** |
   | bias_add | ~8M FLOPs | ~32 MB | **极低**（0.25 FLOP/Byte） |
   | gelu | ~8M FLOPs | ~32 MB | **极低** |

   > `bias_add` 和 `gelu` 的计算几乎可忽略，但各自要读写 ~32MB 数据，白白浪费显存带宽。

   **融合后的变化**：三个操作合并为 1 个 GPU Kernel，matmul 计算完成后结果留在片上高速缓存（寄存器/Shared Memory）中，bias_add 和 gelu 直接在片上操作，不写回 HBM，只有最终结果才写回显存。

   | 对比项 | 未融合（3 个 Kernel） | 融合后（1 个 Kernel） |
   |---|---|---|
   | Kernel 启动次数 | 3 次 | 1 次 |
   | HBM 读写次数 | 6 次（每个算子读+写） | 2 次（开头读 + 结尾写） |
   | 中间结果占用显存 | 需要分配临时 buffer | 不需要 |
   | 额外显存带宽消耗 | ~64 MB（bias_add + gelu 的读写） | 0（在片上完成） |

   **常见融合模式**：

   | 融合模式 | 融合内容 | 典型场景 |
   |---|---|---|
   | Linear + Bias + Activation | matmul → add → gelu/relu/silu | FFN 层 |
   | QKV Projection | 3 个独立 Linear 合并为 1 个大 matmul | Attention 层 |
   | Fused Attention | Q@K → Scale → Mask → Softmax → @V | FlashAttention |
   | LayerNorm + Linear | 归一化后直接矩阵乘 | Transformer 每层入口 |
   | Residual + LayerNorm | 残差连接 + 归一化 | Transformer 每层出口 |

   > **总结**：算子融合 = 把多个独立 Kernel 合并成一个，让中间数据留在片上缓存而不写回显存，消除不必要的显存读写，减少 Kernel 启动开销，本质是"少搬数据"来加速。

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

**什么是 Softmax？**

Softmax 是一个将任意实数向量转换为**概率分布**的函数。给定输入向量 $x = [x_1, x_2, ..., x_n]$，Softmax 的输出为：

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

**核心特性**：
- **输出范围**：每个值都在 (0, 1) 之间
- **归一化**：所有输出之和 = 1，构成合法的概率分布
- **单调性**：输入值越大，对应的输出概率越大
- **放大差异**：由于指数函数的性质，较大的值会被显著放大，较小的值会被压制

**直观理解**：

假设 Attention 计算得到的分数为 $[2.0, 1.0, 0.1]$（表示当前 token 对 3 个 token 的关注程度）：

| 步骤 | Token 1 | Token 2 | Token 3 |
|---|---|---|---|
| 原始分数 | 2.0 | 1.0 | 0.1 |
| $e^{x_i}$ | 7.39 | 2.72 | 1.11 |
| 求和 $\sum$ | 11.22 | 11.22 | 11.22 |
| Softmax | **0.659** | 0.242 | 0.099 |

> Token 1 的分数最高（2.0），经过 Softmax 后获得最大的注意力权重（65.9%），即"重点关注 Token 1"。

**数值稳定性问题**：

直接计算 $e^{x_i}$ 存在溢出风险。当 $x_i$ 很大时（如 FP16 最大值 65504），$e^{x_i}$ 会变成 `inf`。解决方法是减去最大值：

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

减去 $\max(x)$ 后，指数的最大输入为 0（$e^0 = 1$），不会溢出，且数学结果完全等价。

**在大模型中的应用场景**：

| 场景 | 输入 | 作用 |
|---|---|---|
| Attention 层 | $Q \cdot K^T / \sqrt{d}$（score 矩阵） | 将注意力分数转为概率权重 |
| 模型输出层 | logits（词表大小的向量） | 将 logits 转为每个词的生成概率 |
| MoE 路由 | 专家门控分数 | 决定 token 分配给哪些专家 |

**为什么 Softmax 是性能瓶颈？**

以 Attention 中的 Softmax 为例，输入是 $[batch, heads, seq\_len, seq\_len]$ 的矩阵：
- **计算量很小**：只有 exp、sum、div 等逐元素操作
- **数据量很大**：seq_len=4096 时，score 矩阵约 $4096 \times 4096 = 16M$ 个元素
- **是典型的 Memory-bound 操作**：需要对整行数据做 3 次遍历（max → exp+sum → div），每次都要读写 HBM

**标准实现（3-pass）**：
```python
def softmax(x):
    # x: [batch, seq_len]
    # Pass 1: 求 max（数值稳定性）
    # Pass 2: exp 并求 sum
    # Pass 3: 归一化（div）
    exp_x = torch.exp(x - x.max(dim=-1, keepdim=True))
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

**3-pass 的问题**：
- 需要 **3 次遍历整行数据**：max → exp+sum → div，每次都要读写 HBM
- **中间结果** `exp_x` 需要额外显存存储
- 对于 seq_len=4096 的 score 矩阵，3 次 pass 意味着大量显存带宽浪费

**优化：Online Softmax（1-pass）**：
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

**Online Softmax 核心思路**：

标准 Softmax 需要先遍历一次求 max，再遍历一次算 exp+sum，最后遍历一次做 div，共 3 次 pass。Online Softmax 的巧妙之处在于：**在遍历数据的过程中，同时维护 max 和 sum 两个变量**。当遇到更大的值时，利用数学恒等式修正之前累积的 sum：

$$sum_{new} = sum_{old} \times e^{max_{old} - max_{new}} + e^{x_i - max_{new}}$$

这样只需 **1 次 pass** 就能同时得到 max 和 sum，第 2 次 pass 直接计算最终结果。

**优势**：
- 从 3 次 pass 减少到 **2 次 pass**（1 次在线计算 max+sum，1 次输出）
- **不需要存储中间结果** `exp_x`，节省显存
- 是 **FlashAttention 的核心技术**：FlashAttention 将 Q、K、V 分块加载到 SRAM，在片上完成分块 Softmax，再用 Online Softmax 的修正公式合并各块结果，从而避免将完整的 $seq\_len \times seq\_len$ 注意力矩阵写入 HBM

### 4.3 LayerNorm

**什么是 LayerNorm？**

LayerNorm（Layer Normalization，层归一化）是一种**对单个样本在特征维度上做归一化**的技术。简单来说，它把每个 token 的隐藏层表示"拉回"到均值为 0、方差为 1 的标准分布，再通过可学习的缩放参数 $\gamma$（weight）和偏移参数 $\beta$（bias）恢复表达能力。

**公式**：

给定输入向量 $x = [x_1, x_2, ..., x_H]$（H = hidden_size），LayerNorm 的计算过程为：

$$\mu = \frac{1}{H}\sum_{i=1}^{H} x_i \quad \text{（求均值）}$$

$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2 \quad \text{（求方差）}$$

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad \text{（标准化）}$$

$$y_i = \gamma_i \cdot \hat{x}_i + \beta_i \quad \text{（缩放和偏移）}$$

其中 $\epsilon$（一般取 1e-5）是防止除零的小常数。

**直观理解**：

假设 hidden_size=4，某个 token 的隐藏状态为 $[10, -2, 5, 3]$：

| 步骤 | 计算 | 结果 |
|---|---|---|
| 求均值 $\mu$ | $(10 + (-2) + 5 + 3) / 4$ | 4.0 |
| 求方差 $\sigma^2$ | $[(10-4)^2 + (-2-4)^2 + (5-4)^2 + (3-4)^2] / 4$ | 18.5 |
| 标准化 $\hat{x}$ | $(x_i - 4.0) / \sqrt{18.5}$ | $[1.40, -1.40, 0.23, -0.23]$ |
| 缩放偏移 $y$ | $\gamma \cdot \hat{x} + \beta$ | 由学习参数决定 |

> 归一化后，每个 token 的隐藏状态分布被"拉齐"，防止因数值差异过大导致训练不稳定。

**为什么要用 LayerNorm？**

| 问题 | 说明 |
|---|---|
| 内部协变量偏移 | 深度网络中，每层输入的分布会随参数更新不断变化，导致后续层需要反复适应，训练困难 |
| 数值不稳定 | Transformer 有残差连接，层数越深数值累积越大，容易导致梯度爆炸或消失 |
| 收敛速度慢 | 没有归一化时，不同特征的尺度差异大，学习率难以统一设置 |

LayerNorm 通过将每层的输出**标准化到统一的分布**，有效解决了以上问题。

**LayerNorm vs BatchNorm**：

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| **归一化方向** | 沿 batch 维度（跨样本） | 沿 feature 维度（单个样本内） |
| **适用场景** | CNN（图像） | Transformer（NLP/LLM） |
| **batch_size=1** | 统计量不可靠，效果差 | 不受影响，正常工作 |
| **序列长度可变** | 需要 padding 处理 | 天然支持 |
| **推理时** | 需要维护 running_mean/var | 即时计算，无额外状态 |

> 大模型推理中 batch_size 通常较小甚至为 1，且序列长度可变，因此 **LayerNorm 是唯一合适的选择**。

**在大模型中的位置**：

Transformer 中每个层通常有 **2 个 LayerNorm**：
- **Attention 之前/之后**：`LayerNorm → Multi-Head Attention → 残差连接`
- **FFN 之前/之后**：`LayerNorm → Feed-Forward Network → 残差连接`

不同模型的 LayerNorm 放置策略：

| 策略 | 位置 | 代表模型 |
|---|---|---|
| Post-Norm | 残差连接之后 | 原始 Transformer、BERT |
| **Pre-Norm** | 残差连接之前（主流） | GPT-2/3、LLaMA、Qwen |
| RMSNorm | Pre-Norm 的简化版（去掉均值） | LLaMA、Qwen |

> **RMSNorm** 是 LayerNorm 的简化版本，只做方差归一化（不减均值），计算量更少，效果相当：
> $$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H} x_j^2 + \epsilon}} \cdot \gamma_i$$

**为什么 LayerNorm 是性能瓶颈？**

以 hidden_size=4096 为例：
- **计算量极小**：只有求和、求平方、除法等简单操作
- **数据量不小**：每个 token 要读写 4096 个元素，整个序列共 $batch \times seq\_len$ 个 token
- **典型的 Memory-bound 操作**：需要 2 次遍历（第 1 次求 mean+var，第 2 次归一化），瓶颈在显存读写而非计算
- **调用频繁**：每个 Transformer 层调用 2 次，30 层模型就是 60 次 LayerNorm

**标准实现（2-pass）**：
```python
def layer_norm(x, weight, bias, eps=1e-5):
    # x: [batch, seq_len, hidden_size]
    # Pass 1: 求 mean 和 var（需要遍历整个 hidden_size 维度）
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    # Pass 2: 归一化并缩放
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return weight * x_norm + bias
```

**2-pass 的问题**：
- 第 1 次 pass 遍历所有元素计算 mean 和 var
- 第 2 次 pass 再遍历所有元素做归一化
- 中间结果 mean、var 需要额外存储
- 两次 pass 各自产生 HBM 读写，带宽利用率低

**优化：Welford's Online Algorithm（1-pass 计算 mean+var）**：
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

**Welford's Algorithm 核心思路**：

传统方法需要先遍历一次求 mean，再遍历一次求 var（$\sigma^2 = \frac{1}{N}\sum(x_i - \mu)^2$），即 2-pass。Welford 算法的巧妙之处在于：**每读入一个新数据就同时在线更新 mean 和 var**：

```
读入 x_i:
  delta   = x_i - mean_old
  mean    = mean_old + delta / i       # 增量更新均值
  delta2  = x_i - mean_new
  m2      = m2 + delta * delta2        # 增量更新方差的累积量
最终: var = m2 / N
```

**优势**：
- **1 次 pass 即可同时得到 mean 和 var**，配合第 2 次 pass 归一化，总共只需 2 次 pass（传统需要 3 次）
- **数值稳定性好**：不会出现"大数减大数"的精度丢失（传统公式 $\sum x_i^2 - N\mu^2$ 在 FP16 下容易溢出）
- **与 Warp Reduce 结合**：GPU 上每个 warp 的线程各自在线计算局部统计量，最后通过 warp_reduce 合并
- **节省显存带宽**：减少了一次完整的 HBM 读取，对于 hidden_size=4096 的典型场景，带宽节省约 33%

> **一句话总结**：LayerNorm 本身计算简单，但因为调用极其频繁（每层 2 次），优化的关键是**减少 HBM 读写次数**。Welford 算法通过在线计算将 pass 数从 3 减到 2，是 GPU 上 LayerNorm kernel 的标准实现方式。

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

> 💼 **实战关联**：
> - 你在 nsys 报告里看到的 `flash_fwd_kernel` 就是 FlashAttention 的 Prefill kernel，`flash_bwd_kernel` 是反向传播用的
> - Online Softmax 是 FlashAttention 的核心技术之一——理解了它，就理解了为什么 FlashAttention 能不存储完整的 N² 注意力矩阵
> - Welford's Algorithm 是所有主流推理框架中 LayerNorm kernel 的标准实现，你在 CUTLASS/Triton 源码里都能看到
> - 算子融合（Fusion）是最容易获得"免费"加速的优化——vLLM/SGLang 已经默认启用了大部分常见融合模式

---

## 5. 性能分析与调优

> 🟢 **一句话版本**：不 Profile 就优化等于蒙眼开车。用 Nsight Systems 看全局时间线（找到最慢的 kernel），用 Nsight Compute 深入分析单个 kernel（看它是卡在算力还是带宽），用 PyTorch Profiler 从 Python 层面定位问题。三层工具配合使用，才能精准定位瓶颈。

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

> 💼 **实战关联**：
> - 实际优化工作流：先用 `nsys profile` 采集全局 timeline → 找到最耗时的 kernel → 用 `ncu` 深入分析该 kernel 的瓶颈 → 对症下药
> - vLLM 中可以通过 `--enforce-eager` 禁用 CUDA Graph 来方便调试和 profiling
> - 当 nsys 显示大量小 kernel 密集排列时，优先考虑 CUDA Graph 或算子融合
> - 当某个 kernel 的 Memory Throughput 接近硬件峰值但 Compute 利用率很低时，说明是 Memory-bound，考虑量化或减少数据搬运

---

## 🎯 速查卡片

> 以下是全文核心知识的一页浓缩，可以打印贴工位方便随时查阅。

### GPU 硬件速查

| 概念 | 一句话记忆 | 关键数字 |
|------|-----------|----------|
| **SM** | GPU 的基本计算单元，包含 CUDA Core + Tensor Core | A100: 108个, H100: 132个 |
| **Tensor Core** | 专用矩阵乘法硬件，比 CUDA Core 快 16x | 需要 FP16/BF16/FP8 + 维度对齐 |
| **Warp** | 32 个线程的执行单元 | GPU 调度的最小粒度 |
| **HBM** | GPU 显存，3D 堆叠高带宽 | A100: 2 TB/s, H100: 3.35 TB/s |
| **SRAM** | 片上高速缓存 | 192 KB/SM，比 HBM 快 10x |
| **Registers** | 最快的存储 | 1 cycle 延迟，~20 TB/s |

### 内存层次速查

| 层级 | 延迟 | 容量 | 带宽 | 类比 |
|------|------|------|------|------|
| Registers | 1 cycle | ~KB/SM | ~20 TB/s | 手边便签纸 |
| SRAM/L1 | ~20 cycles | 192 KB/SM | ~19 TB/s | 桌上笔记本 |
| L2 Cache | ~200 cycles | 40 MB | ~7 TB/s | 抽屉里的文件 |
| HBM | ~300-400 cycles | 40-80 GB | 1.6-3.35 TB/s | 楼下仓库 |

### 关键算子速查

| 算子 | 一句话记忆 | 优化核心 | 关键技术 |
|------|-----------|---------|----------|
| **MatMul** | 大模型最耗时的运算 | 用 Tensor Core + 分块 | cuBLAS, CUTLASS |
| **Softmax** | 概率归一化，3-pass 太慢 | 减少 pass 数 | Online Softmax（1-pass） |
| **LayerNorm** | 归一化层，调用极频繁 | 减少 HBM 读写 | Welford's Algorithm |
| **FlashAttention** | Attention 的终极优化 | 分块 + 不存 N² 矩阵 | 在线 Softmax + SRAM 分块 |
| **算子融合** | 多个 kernel 合一个 | 中间结果留在片上 | Linear+Bias+Activation 融合 |

### Profiling 工具速查

| 工具 | 定位 | 看什么 | 命令 |
|------|------|--------|------|
| **nsys** | 全局 timeline | 哪个 kernel 最慢、GPU 空闲在哪 | `nsys profile -o out python xx.py` |
| **ncu** | 单 kernel 深入 | SM 利用率、带宽、Occupancy | `ncu --set full -o out python xx.py` |
| **PyTorch Profiler** | Python 层面 | 算子耗时、内存分配 | `with torch.profiler.profile(...)` |

### 性能诊断速查

| 现象 | 可能原因 | 对应概念 | 解决方向 |
|------|---------|---------|----------|
| GPU 利用率 <20% | Memory-bound（Decode 阶段） | §2 HBM 带宽 | 量化、GQA/MQA |
| 大量小 kernel | Kernel Launch Overhead | §5.3 | CUDA Graph、算子融合 |
| Tensor Core 利用率 0% | 数据类型/对齐不满足 | §1.3 Tensor Core | FP16/BF16 + 维度对齐 |
| Warp Occupancy 低 | Register/SMEM 用量过大 | §1.3 SM 架构 | 调整 Block Size |
| Shared Memory 性能差 | Bank Conflict | §2.4 | Padding (`[N][N+1]`) |

---

## 总结

### 核心要点

1. **GPU 架构**：
   - 高并行度（数千核心）
   - 高内存带宽（2+ TB/s）
   - Tensor Core（16x 加速）

2. **内存层次**：
   - Registers > Shared Memory > L2 Cache > HBM
   - HBM：3D 堆叠 + 硅中介层，高带宽（2-4.8 TB/s），是 GPU 显存主体
   - Decode 阶段是 Memory-bound，HBM 带宽是生成速度上限
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

> 📖 **相关阅读**：如果你还没看 Part1，建议先阅读 [大模型推理技术分享-Part1-基础概念](./大模型推理技术分享-Part1-基础概念.md)，了解 KV Cache、Prefill/Decode、FlashAttention 等核心推理概念。
