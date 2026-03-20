# Nsight Systems (nsys) 性能分析实战指南 — vLLM 推理场景

## 目录

- [1. nsys 简介与数据采集](#1-nsys-简介与数据采集)
- [2. 时间线分析（Timeline View）](#2-时间线分析timeline-view)
  - [2.5 主引擎核心线程详解](#25-主引擎核心线程详解)
  - [2.6 Stream 与 Memory 操作详解](#26-stream-与-memory-操作详解)
  - [2.7 多线程结构与 cudaEventSynchronize 分析](#27-多线程结构与-cudaeventsynchronize-分析)
- [3. CUDA GPU Kernel Summary 分析](#3-cuda-gpu-kernel-summary-分析)
- [4. CUDA API Summary 分析](#4-cuda-api-summary-分析)
- [5. CUDA Kernel Launch & Exec Time 分析](#5-cuda-kernel-launch--exec-time-分析)
- [6. CUDA GPU Trace 分析](#6-cuda-gpu-trace-分析)
- [7. CUDA GPU Kernel/Grid/Block Summary 分析](#7-cuda-gpu-kernelgridblock-summary-分析)
- [8. CUDA Graph 原理与分析](#8-cuda-graph-原理与分析)
- [9. 优化建议总结](#9-优化建议总结)
- [10. 常用操作备忘](#10-常用操作备忘)

---

## 1. nsys 简介与数据采集

### 1.1 什么是 nsys

NVIDIA Nsight Systems (nsys) 是 NVIDIA 提供的系统级性能分析工具，可以对 GPU 应用进行全面的时间线跟踪，
包括 CUDA kernel 执行、内存操作、API 调用、线程活动等。

### 1.2 数据采集命令

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  -o vllm_nsys_report \
  <your_vllm_command>
```

**关键参数说明**：

| 参数 | 说明 |
|------|------|
| `--trace=cuda` | 采集 CUDA kernel、内存操作、API 调用（**必须**） |
| `--trace=nvtx` | 采集 NVTX 标注信息（vLLM 支持，可看到 prefill/decode 各阶段耗时） |
| `--trace=osrt` | 采集操作系统运行时信息 |
| `--cuda-memory-usage=true` | 记录 CUDA 内存使用情况 |
| `-o` | 输出报告文件名 |

### 1.3 离线查看 Kernel Summary

如果无法在 GUI 中操作，可以使用命令行导出 Kernel Summary：

```bash
nsys stats -r cuda_gpu_kern_sum "/path/to/vllm_nsys_report.nsys-rep"
```

---

## 2. 时间线分析（Timeline View）

### 2.1 分析维度

打开 nsys 报告后，Timeline View 主要关注以下几行：

| 行 | 含义 | 关注点 |
|----|------|--------|
| **CUDA HW** | GPU 硬件实际执行情况 | 观察 GPU 是否有空闲间隙（气泡） |
| **[All Streams]** | 所有 CUDA stream 的聚合视图 | 区分 Graphs / Kernels / Memory 占比 |
| **Stream N** | 各个 CUDA stream 的独立视图 | 区分计算 stream 和通信 stream |
| **Threads** | CPU 线程活动 | 判断 CPU 侧是否有瓶颈 |

### 2.2 实际分析案例

对一个 vLLM 服务的分析结果（**GPU: NVIDIA A100-SXM4-40GB**，单卡、TP=1）：

![nsys 时间线总览](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/b4c51dbe-40a7-4104-bd09-26a5844a4b34/image-019d040c60cb700089314f33b0a72f83-019d09da-bfac-705c-87cb-171ef6f8ea6f.png)

| 指标 | 占比 | 解读 |
|------|------|------|
| **Graphs（CUDA Graph）** | 62.1% | 占 GPU 执行时间的大头，说明 vLLM 的 CUDA Graph capture 生效了 |
| **Kernels（eager 模式）** | 36.7% | 非 Graph 方式直接 launch 的 kernel，主要来自 prefill 阶段 |
| **Memory** | 1.1% | 显存操作占比极低，说明没有频繁的 Host↔Device 数据搬运 |

**Stream 分布**：

| Stream | 占比 | 说明 |
|--------|------|------|
| **Default stream 7** | 99.8% | 承载几乎所有计算（Graphs 62.2% + Kernels 36.7% + Memory） |
| **Stream 17** | 0.2% | 100% DtoH memcpy — 专用于将采样结果从 GPU 回传到 CPU |

### 2.3 GPU 气泡（Idle Gap）分析

**现象**：在时间线上观察到 **100~200ms 的 GPU 空闲间隙**

**可能原因**：
1. **vLLM 调度器（Scheduler）在做 batch 调度决策** — 等待新请求或重新调度 sequences
2. **Prefill → Decode 阶段切换** — chunked prefill 切换阶段的 CPU 侧开销
3. **Python GIL / CPU 侧瓶颈** — 调度器是 Python 实现的，可能有 CPU-bound 延迟

**优化方向**：增加并发请求量，让 scheduler 有更多 sequence 可以调度，填满 GPU

### 2.4 线程分析

主要计算集中在主引擎线程（如 `VLLM::EngineCor`），其他线程活动较少，符合 `tensor-parallel-size=1` 的配置特征。
通信 stream 几乎无活动（仅占 0.2%），进一步验证了单卡部署无通信瓶颈。

### 2.5 主引擎核心线程详解

在 nsys 时间线上，主引擎核心线程（通常标注为 `VLLM::EngineCor` 或类似名称）是 vLLM 推理引擎中**最关键的线程**。
它包含 CCCL 活动、CUDA API 调用和 Profiler Overhead（绿色块）三类关键元素。

#### 2.5.1 主引擎线程的职责

```
┌───────────────── 主引擎线程的职责 ─────────────────┐
│                                                     │
│  1. Scheduler 调度 → 决定哪些请求进入当前 batch     │
│  2. 模型 Forward → 触发 GPU 上的所有计算            │
│  3. Sampler 采样 → 从 logits 中选出下一个 token     │
│  4. KV Cache 管理 → 分配/释放 cache block           │
│  5. 结果回传 → 将生成的 token 返回给调用方           │
│                                                     │
│  ← 这一切都在同一个线程中串行执行                    │
└─────────────────────────────────────────────────────┘
```

#### 2.5.2 CCCL（CUDA C++ Core Libraries）

CCCL 是 NVIDIA 的 **CUDA C++ 核心库**，包含以下子库：

| 库 | 全称 | 功能 |
|----|------|------|
| **Thrust** | CUDA 并行算法库 | 提供类似 C++ STL 的并行 sort、reduce、scan 等 |
| **CUB** | CUDA Unbound | 底层 GPU 原语：block-level reduce、scan、radix sort 等 |
| **libcudacxx** | CUDA C++ 标准库 | GPU 上的 C++ 标准库实现（atomic、barrier 等） |

**在 vLLM 中触发 CCCL 的场景**：

| 场景 | 使用的 CCCL 功能 | 说明 |
|------|-----------------|------|
| Paged Attention 的 block 索引管理 | CUB 的 DeviceScan/DeviceSelect | 对 block table 做前缀和/筛选 |
| Top-k / Top-p 采样 | CUB/Thrust 的排序和归约 | 从 logits 中选取候选 token |
| Token ID 的 gather/scatter | Thrust 的 gather/scatter | 从离散的 KV cache block 中收集/分发数据 |
| 各类辅助计算 | CUB 前缀和 | cumsum、argmax、tensor 重排等 |

对应到 Kernel Summary 中，`computeDigitCumSum`（34292 次调用、平均 6.3µs）就是典型的 CUB 前缀和操作。

#### 2.5.3 CUDA API 调用

时间线上标注的 CUDA API 调用，是 **CPU 侧**通过 CUDA Runtime/Driver API 与 GPU 交互的操作。

**Kernel Launch 相关**：

| API | 说明 | 频率 |
|-----|------|------|
| `cudaLaunchKernel` | 提交单个 kernel 到 GPU 执行（eager 模式） | 高频 |
| `cudaGraphLaunch` | 回放整张 CUDA Graph（decode 阶段） | 中频 |
| `cudaStreamSynchronize` | 等待 GPU stream 上所有操作完成 | 低频 |

**内存管理相关**：

| API | 说明 | 场景 |
|-----|------|------|
| `cudaMallocAsync` | 异步分配 GPU 显存 | KV Cache block 分配 |
| `cudaFreeAsync` | 异步释放 GPU 显存 | KV Cache block 回收 |
| `cudaMemcpyAsync` | 异步内存拷贝（Host↔Device） | 传输 input_ids、采样参数等 |
| `cudaMemsetAsync` | 异步内存填充 | 初始化 buffer |

**同步与事件相关**：

| API | 说明 | 场景 |
|-----|------|------|
| `cudaEventRecord` | 在 stream 上记录一个事件 | 标记时间点 |
| `cudaEventQuery` | 查询事件是否完成（非阻塞） | 异步检查 GPU 进度 |
| `cudaStreamWaitEvent` | 让一个 stream 等待另一个 stream 的事件 | 多 stream 同步 |

**关键点**：时间线上的 CUDA API 块表示的是 **CPU 侧提交命令的时间**，而非 GPU 实际执行的时间。两者是异步的——CPU 提交后通常不会等待 GPU 完成（除非显式同步）。

#### 2.5.4 Profiler Overhead（绿色块）

时间线上的**绿色块是 nsys 本身引入的测量开销**，不是应用代码产生的。

**Profiler Overhead 的来源**：

| 来源 | 说明 |
|------|------|
| 拦截 CUDA API 调用 | 在 API 入口/出口注入 hook，记录时间戳、参数、调用栈 |
| 采集 GPU kernel 信息 | 通过 CUPTI（CUDA Profiling Tools Interface）监听 kernel launch |
| 缓冲区管理 | 采集数据先写入环形缓冲区，缓冲区满时 flush 到磁盘/内存，可能短暂阻塞被监控线程 |
| 线程上下文切换记录 | 追踪操作系统层面的调度信息 |

**对分析的影响**：

| 影响 | 说明 |
|------|------|
| **时间膨胀** | 总时间比实际运行时长 **2-5%** |
| **相对比例基本可信** | 各 kernel 之间的**相对占比**基本不受影响 |
| **不影响定性结论** | kernel 排行、GPU 利用率趋势等定性结论仍然成立 |

**减少 Profiler Overhead 的方法**：

```bash
# 只采集 CUDA 相关，减少采集范围
nsys profile --trace=cuda -o report <command>

# 采样模式代替完整跟踪（开销更小）
nsys profile --sample=cpu --trace=cuda -o report <command>

# 限制采集时长，避免产生过大的数据量
nsys profile --trace=cuda --duration=10 -o report <command>

# 关闭 backtrace 采集（减少 CPU 端开销）
nsys profile --trace=cuda --backtrace=none -o report <command>
```

#### 2.5.5 主引擎线程时间线总览

```
     主引擎核心线程 (VLLM::EngineCor)
     ─────────────────────────────────────────────────────────
时间 →

     ┌─────┐┌──┐┌─────────┐┌──┐┌────────────┐┌──┐┌─────┐
     │Sched││MC││ CUDA API ││PR││ CUDA Graph ││PR││Sched│
     │uler ││py││ Launch   ││OH││  Launch    ││OH││uler │
     │(CPU)││  ││ (eager)  ││  ││ (decode)   ││  ││(CPU)│
     └─────┘└──┘└─────────┘└──┘└────────────┘└──┘└─────┘

     ├─────────── prefill ──────────┤├── decode ──┤

     MCpy = cudaMemcpyAsync         (蓝色/橙色块)
     PROH = Profiler Overhead       (绿色块)
     CCCL = CUB/Thrust 原语调用     (穿插在 CUDA API 中)
```

| 颜色/类别 | 含义 | 关注什么 |
|-----------|------|----------|
| **CUDA API 块** | CPU 提交 GPU 命令的时间 | 如果占比过高，说明 CPU 成为瓶颈 |
| **CCCL 活动** | CUB/Thrust 库的调用 | 通常是辅助计算，不应该是主要开销 |
| **绿色 Profiler Overhead** | nsys 自身的采集开销 | **忽略它**，不是应用问题 |
| **空白间隙** | CPU 侧处理（Python 调度等） | 如果间隙很长，说明调度或 Python GIL 有问题 |

### 2.6 Stream 与 Memory 操作详解

#### 2.6.1 Stream 架构

vLLM 使用 **多 stream 架构**将计算与数据传输分离：

```
┌─────────────────────────────────────────────────────┐
│  Default stream 7 (99.8%)                           │
│  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │  Graphs   │  │  Kernels  │  │    Memory       │  │
│  │  62.2%    │  │  36.7%    │  │    1.1%         │  │
│  │ (decode)  │  │ (prefill) │  │ (HtoD/Memset等) │  │
│  └───────────┘  └───────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Stream 17 (0.2%)                                   │
│  ┌─────────────────────────────────────────────┐    │
│  │  100% DtoH memcpy — 采样结果回传专用        │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**关键点**：
- **几乎所有计算都在单一 stream 上** — 符合 TP=1 单卡部署特征
- **Stream 17 专用于 DtoH 回传** — vLLM 将采样结果回传放在独立 stream，避免阻塞主计算 stream
- **通信 stream 无活动** — 验证无多卡通信开销

#### 2.6.2 Memory 操作细分

从时间线中可以看到 Memory 操作（占 GPU 时间 1.1%）的详细分布：

| 类型 | 占 Memory 比例 | 说明 |
|------|---------------|------|
| **HtoD memcpy** | 84.5% | Host→Device 拷贝，主要是 input_ids、位置编码、采样参数等传输到 GPU |
| **Memset** | 8.9% | 显存填充/清零，通常用于初始化 buffer 或清空缓存 |
| **DtoD memcpy** | 5.4% | Device→Device 拷贝，GPU 内部的数据搬运（如 KV cache 的 block 拷贝/整理） |
| **DtoH memcpy** | 1.2% | Device→Host 拷贝，将生成的 token ID、logits 等结果回传 CPU |

**分析要点**：
- **HtoD 占绝对主导（84.5%）** — 这是正常的，因为每个 step 都需要将新的 input 传到 GPU
- **DtoD 占 5.4%** — 来自 KV cache 的 block 管理操作（Paged Attention 需要在不同 block 间搬运数据）
- **DtoH 很少（1.2%）** — 只回传 token ID 等极少量数据，且放在独立 stream 17 异步执行
- **Memory 总占比仅 1.1%** — 说明数据传输不是性能瓶颈

### 2.7 多线程结构与 cudaEventSynchronize 分析

#### 2.7.1 vLLM 线程结构

从时间线中可以观察到 vLLM 使用了 **4 个线程**（`Threads (4)`），均标注为 `VLLM::EngineCor`：

| 线程 | TID | 角色 | 包含的活动 |
|------|-----|------|------------|
| **[592] VLLM::EngineCor** | 592 | 主计算线程 | CCCL + CUDA API + Profiler overhead |
| **[1377] VLLM::EngineCor** | 1377 | 辅助线程 | CUDA API + Profiler overhead（活动较少） |
| **[1907] VLLM::EngineCor** | 1907 | 同步/输出线程 | CUDA API（主要是 cudaEventSynchronize） + Profiler overhead |
| 其他线程 | - | 后台线程 | 少量活动 |

```
线程 [592]  ─── CCCL ──── CUDA API (kernel launch) ──── CCCL ────
                 ▲ 主要负责提交 GPU 计算任务

线程 [1377] ─── ··· ─── (少量 CUDA API) ─── ··· ───
                 ▲ 辅助任务

线程 [1907] ─── ▓▓▓ cudaEventSynchronize ▓▓▓ ─── CUDA API ───
                 ▲ 等待 GPU 完成，然后处理输出
```

#### 2.7.2 cudaEventSynchronize 深度分析

**Events View** 中可以看到大量的 `cudaEventSynchronize` 调用（均在 TID 1907 上）：

| 调用序号 | 开始时间 | 耗时 | 说明 |
|---------|---------|------|------|
| #1 | 0.215s | **65.885 ms** | 🔴 首次调用，耗时极长 — 模型 warmup / 首次 CUDA Graph capture |
| #2 | 0.285s | 1.121 ms | 正常范围 |
| #3 | 0.368s | 24.221 ms | 🟡 偏长 — 可能是 prefill 大请求 |
| #4~#13 | 0.394s+ | **3.5~3.8 ms** | ✅ 稳定 — 正常 decode 步骤间隔 |

**`cudaEventSynchronize` 是什么？**

它是一个 **阻塞式 CPU 同步调用**：CPU 线程会阻塞等待，直到 GPU 上指定的 event 完成。

```
CPU 线程 [1907]:  ──── cudaEventSynchronize(event) ────
                       │  ← CPU 阻塞等待              │
                       │                               │
GPU:              ═══ kernel A ═══ kernel B ═══ done! ══╡
                                                        │
                       CPU 被唤醒，继续执行 ─────────────┘
```

**在 vLLM 中的作用**：
- 线程 [1907] 通过 `cudaEventSynchronize` 等待当前 step 的 GPU 计算完成
- GPU 完成后，CPU 可以读取采样结果（token ID），进行输出流式推送
- 然后主线程可以开始下一个 step 的调度

**分析要点**：

| 现象 | 原因 | 是否正常 |
|------|------|----------|
| 首次调用 65.885ms | 模型首次推理 / CUDA Graph capture / CUDA 上下文初始化 | ✅ 正常 |
| 稳定在 3.5~3.8ms | 正常 decode step 的 GPU 计算时间 | ✅ 正常 |
| 偶尔出现 24ms | 可能是 prefill 大请求穿插 | ⚠️ 关注 |
| 如果持续 >10ms | GPU 计算效率下降或 batch 过大 | 🔴 需排查 |

**`cudaEventSynchronize` 的耗时本质上反映了 GPU 侧的计算时间**：
- 3.5ms ≈ 一次 decode step 中所有 GPU kernel 的总执行时间
- 这与 Kernel Summary 中各 kernel 平均耗时的累加基本一致
- 如果此值持续偏高，说明 GPU 计算本身偏慢（可能是 batch 过大或模型过大）

#### 2.7.3 NVTX (CCCL) 标注行

在 Memory 行下方可以看到 **NVTX (CCCL)** 行，这是 NVIDIA CCCL 库自带的 NVTX 标注，
用于标记 CUB/Thrust 等库函数的执行区间。结合 `--trace=nvtx` 参数可以在时间线上看到具体的 CCCL 调用范围。

---

## 3. CUDA GPU Kernel Summary 分析

### 3.1 如何查看

在 Nsight Systems GUI 左下角的 **Stats System View** 列表中，单击 **"CUDA GPU Kernel Summary"**，右侧面板会展示 kernel 耗时排行。

**推荐查看的报告**：

| 报告名称 | 用途 |
|----------|------|
| **CUDA GPU Kernel Summary** | 查看哪些 GPU kernel 耗时最多，按总时间排序 |
| **CUDA GPU Trace** | 查看每个 kernel 的详细执行时间线 |
| **CUDA Kernel Launch & Exec Time Summary** | 查看 kernel launch 延迟（CPU 提交到 GPU 实际开始执行的间隔） |
| **CUDA API Summary** | 查看 cudaMalloc/cudaMemcpy 等 API 调用的耗时分布 |

### 3.2 Kernel 分类与耗时分布

以一个实际 vLLM 推理服务（BF16 精度、Ampere 架构 GPU）的 Kernel Summary 为例：

![CUDA GPU Kernel Summary](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/c7a963c5-4b6c-423d-ae9f-aa8901d8600a/image-019d040c60cb700089314f33b0a72f83-019d09fc-10cf-76f0-a8d2-79a6deab0568.png)

#### 3.2.1 GEMM（矩阵乘法）— 占比约 56%

| 排名 | 占比 | 总耗时 | 调用次数 | 平均耗时 | Kernel 名称 |
|------|------|--------|----------|----------|-------------|
| 1 | 17.3% | 3.730s | 8579 | 434.8µs | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` |
| 2 | 16.0% | 3.470s | 8848 | 392.1µs | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn` |
| 4 | 11.2% | 2.422s | 7500 | 322.9µs | `ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_32x5_tn` |
| 5 | 7.5% | 1.620s | 5177 | 313.0µs | `ampere_bf16_s16816gemm_bf16_128x256_ldg8_relu_f2f_stages_64x3_tn` |
| 7 | 3.5% | 765.7ms | 3024 | 253.2µs | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn` |
| 8 | 2.1% | 449.2ms | 1876 | 239.4µs | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn` |

**分析要点**：
- 使用 `ampere_bf16_s16816gemm` 系列 kernel，说明在 Ampere 架构 GPU 上使用了 **BF16 Tensor Core**
- 出现多种 tile size（64x64、128x256、128x128、256x128），说明 cuBLAS 自动选择了不同形状的 GEMM 配置
- **没有看到 FP8 kernel**，如果 GPU 支持（如 H100/H800），可以考虑启用 FP8 量化进一步提速

#### 3.2.2 Flash Attention — 占比约 18.4%

| 排名 | 占比 | 总耗时 | 调用次数 | 平均耗时 | Kernel 名称 |
|------|------|--------|----------|----------|-------------|
| 3 | 11.5% | 2.493s | 4060 | 614.1µs | `flash_fwd_splitkv_kernel` (decode 阶段) |
| 6 | 6.9% | 1.489s | 3264 | 456.1µs | `flash_fwd_kernel` (prefill 阶段) |

**分析要点**：
- `flash_fwd_splitkv_kernel` 是 **decode 阶段**的 split-KV attention，耗时占比更高说明采集期间 decode 请求占比更多
- `splitkv_kernel` 平均耗时 614µs，是所有 kernel 中**单次调用最慢的**，与 KV cache 长度直接相关
- 如果 `max-model-len` 配置较大（如 38400），会显著增加 splitkv attention 的耗时

#### 3.2.3 激活函数与归一化 — 占比约 6%

| 占比 | Kernel | 说明 |
|------|--------|------|
| 2.0% | `GeluCUDAKernelImpl` | GELU 激活函数 |
| 2.0% | `cunn_SoftMaxForward` | Softmax |
| 1.3% | `triton_poi_fused_mul_silu_slice_1` | SwiGLU 激活函数（Triton 自定义算子） |
| 0.7% | `triton_red_fused_to_copy_add_mean_mul_pow_rsqrt_2` | RMSNorm（Triton 自定义算子） |

#### 3.2.4 碎片化小 Kernel — 存在 launch overhead

| 占比 | 调用次数 | 平均耗时 | Kernel | 说明 |
|------|----------|----------|--------|------|
| 1.7% | 4795 | 78.3µs | `elementwise_kernel` | 逐元素操作 |
| 1.6% | 7072 | 49.7µs | `vectorized_layer_norm_kernel` | LayerNorm |
| 1.1% | 3264 | 70.0µs | `rotary_kernel` | RoPE 旋转位置编码 |
| 1.0% | 34292 | 6.3µs | `computeDigitCumSum` | 前缀和计算 |

**分析要点**：
- 这些小 kernel 单次执行时间很短（几 µs 到几十 µs），但调用次数多
- 容易导致 **kernel launch overhead**
- CUDA Graph 可以有效缓解此问题

#### 3.2.5 vLLM 框架 Kernel

| 占比 | Kernel | 说明 |
|------|--------|------|
| 0.7% | `reshape_and_cache_flash_kernel` | vLLM 的 KV Cache 管理 kernel，负责把新生成的 KV 写入 Paged Attention 的 cache block |

### 3.3 Kernel 分析综合评估

| 维度 | 评估 | 说明 |
|------|------|------|
| **计算效率** | ✅ 良好 | GEMM 使用了 BF16 Tensor Core |
| **Attention** | ⚠️ 可关注 | splitkv_kernel 单次 614µs 偏高，可能因 max-model-len 过大 |
| **Kernel Launch** | ⚠️ 碎片化 | 大量小 kernel（cumsum、elementwise 等调用 3 万+ 次） |
| **量化** | ❌ 未启用 | 没有 FP8/INT8 kernel，当前全 BF16 推理 |
| **CUDA Graph** | ✅ 已启用 | 62% 被 CUDA Graph 覆盖，有助于减少 launch overhead |

---

## 4. CUDA API Summary 分析

CUDA API Summary 汇总了 CPU 侧所有 CUDA API 调用的统计信息，是判断 **CPU 侧是否存在瓶颈**的关键报告。

![CUDA API Summary](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/726494b6-7db2-4ada-aab2-59540f1cd283/image-019d040c60cb700089314f33b0a72f83-019d0a29-02a6-7fda-8fa8-66d908bae987.png)

### 4.1 API 调用耗时排行

| 排名 | 占比 | 总耗时 | 调用次数 | 平均耗时 | API 名称 | 说明 |
|------|------|--------|----------|----------|----------|------|
| 1 | **78.1%** | 51.176s | 25743 | 1.988ms | `cudaEventSynchronize` | CPU 阻塞等待 GPU 完成 |
| 2 | 10.4% | 6.804s | 3400 | 2.001ms | `cudaStreamSynchronize` | 等待 stream 上所有操作完成 |
| 3 | 6.0% | 3.900s | 529971 | 7.358µs | `cudaLaunchKernel` | eager 模式 kernel 提交 |
| 4 | 2.2% | 1.436s | 124797 | 11.510µs | `cudaMemcpyAsync` | 异步内存拷贝 |
| 5 | 1.6% | 1.016s | 9073 | 112.014µs | `cudaGraphLaunch_v10000` | CUDA Graph 回放 |
| 6 | 0.4% | 287.3ms | 37933 | 7.574µs | `cuLaunchKernel` | Driver API kernel 提交 |
| 7 | 0.3% | 268.4ms | 51827 | 5.177µs | `cudaEventRecordWithFlags` | 事件记录 |
| 8 | 0.3% | 189.3ms | 77457 | 2.444µs | `cudaEventQuery` | 非阻塞查询事件状态 |
| 9 | 0.2% | 157.6ms | 27445 | 5.741µs | `cudaMemsetAsync` | 异步内存填充 |
| 10 | 0.1% | 74.8ms | 69480 | 1.076µs | `cudaStreamIsCapturing` | 查询 stream 是否在 Graph 录制中 |
| 其他 | <0.1% | - | - | - | `cudaEventDestroy` / `cudaMalloc` / `cudaHostAlloc` 等 | 低频操作 |

### 4.2 关键发现与分析

#### 4.2.1 同步等待占绝对主导 — 88.5%

`cudaEventSynchronize`（78.1%）+ `cudaStreamSynchronize`（10.4%）= **88.5%**。
这意味着 CPU 主线程 88.5% 的时间在**等待 GPU 完成计算**。

**解读**：
- 这**不是异常**，而是 GPU 密集计算场景的典型特征 —— CPU 提交任务后，大部分时间在等 GPU
- 如果同步占比 <50%，反而说明 CPU 侧有太多工作要做，成为了瓶颈
- `cudaEventSynchronize` 的平均耗时 1.988ms ≈ 一次 decode step 的 GPU 计算时间
- `cudaStreamSynchronize` 的平均耗时 2.001ms，调用次数（3400 次）远少于 Event 同步（25743 次）

#### 4.2.2 Kernel Launch 效率分析

| Launch 方式 | 调用次数 | 平均耗时 | 总耗时 | 说明 |
|------------|----------|----------|--------|------|
| `cudaLaunchKernel` | 529,971 | 7.358µs | 3.900s | eager 模式，prefill 阶段大量使用 |
| `cuLaunchKernel` | 37,933 | 7.574µs | 287ms | Driver API 级别的 kernel launch |
| `cudaGraphLaunch` | 9,073 | **112.014µs** | 1.016s | CUDA Graph 回放 |

**分析要点**：
- `cudaLaunchKernel` 平均 7.4µs，对于 eager 模式这是正常水平
- `cudaGraphLaunch` 平均 112µs，虽然单次比 eager launch 慢，但它**一次执行整张 Graph（几百个 kernel）**，整体效率远高于逐个 launch
- 529,971 次 eager launch vs 9,073 次 graph launch，比例约 **58:1**，与 Kernel 37% vs Graph 62% 的 GPU 时间占比吻合

#### 4.2.3 异步操作分析

| 操作 | 调用次数 | 说明 |
|------|----------|------|
| `cudaMemcpyAsync` | 124,797 | 大量异步 memcpy，主要是 HtoD 传输 input 数据 |
| `cudaEventQuery` | **77,457** | vLLM 频繁轮询 GPU 是否完成（非阻塞），是异步调度机制的体现 |
| `cudaEventRecordWithFlags` | 51,827 | 记录事件标记，用于后续同步 |
| `cudaStreamIsCapturing` | 69,480 | 查询 stream 是否在 CUDA Graph capture 模式中 |
| `cudaMemsetAsync` | 27,445 | 内存填充/清零 |

**分析要点**：
- `cudaEventQuery` 调用 7.7 万次，平均仅 2.4µs，是**非阻塞的轮询**操作。vLLM 通过它来异步检查 GPU 进度，避免不必要的阻塞等待
- `cudaStreamIsCapturing` 被调用近 7 万次，说明 vLLM 内部频繁判断当前是否处于 Graph capture 状态，以决定走 eager 还是 graph 路径

#### 4.2.4 内存分配

表尾可以看到 `cudaMalloc`（768µs，仅 1 次）和 `cudaHostAlloc`（354µs，12 次）。
- 只有 1 次 `cudaMalloc` — vLLM 在启动时**一次性预分配**了所有 GPU 显存（KV Cache Pool），运行中不再动态分配
- 12 次 `cudaHostAlloc` — 分配 Host pinned memory，用于高速 HtoD/DtoH 传输

---

## 5. CUDA Kernel Launch & Exec Time 分析

这组报告展示了从 CPU launch 到 GPU 实际执行完毕的**全链路延迟**，是分析 kernel 调度效率的核心工具。

### 5.1 Launch & Exec Time Summary

![CUDA Kernel Launch & Exec Time Summary](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/8124cc24-6f3c-4b7c-a4fa-7076de26d977/image-019d040c60cb700089314f33b0a72f83-019d0a29-d034-74cd-9756-6ec6dda869ee.png)

按 kernel 类型汇总的 launch 延迟统计：

| 关键指标 | 数值 | 说明 |
|---------|------|------|
| **PID/TID** | 全部 592/592 | 确认所有 kernel launch 都在单一线程上 |
| **Count=571, QCount=568** | 571 次 launch，568 次有排队 | 99.5% 的 kernel 都经历了排队，说明 GPU 一直在忙 |
| **TAvg（总平均延迟）** | 32.360ms | 从 CPU launch 到 GPU 执行完毕的平均全链路时间 |
| **KAvg（Kernel 平均执行时间）** | 11.597µs ~ 434.822µs | GPU 侧实际计算耗时 |
| **QAvg（排队平均等待时间）** | 8.289µs ~ 25.353ms | 在 GPU 命令队列中的等待时间 |
| **AMax（API 最大耗时）** | 276.580ms | 某次 API 调用耗时极高（可能是 prefill 大请求） |

**关键发现**：

1. **Queue 等待时间普遍在 ms 级别** — 这说明 CPU 的提交速度快于 GPU 的消费速度，GPU 一直在忙碌，这是健康状态
2. **TMax（Total Maximum）高达 226~448ms** — 极端情况下从 launch 到完成耗时很长，主要是 prefill 大请求导致前面排队的 kernel 要等很久
3. **KAvg vs QAvg** — Kernel 实际执行时间（µs 级）远小于排队时间（ms 级），说明性能瓶颈不在 kernel 本身，而在整体 pipeline 的串行等待上
4. **最后一行** Count=8579，KAvg=434.822µs — 这是调用最频繁的 GEMM kernel，单次执行 434.8µs，与 Kernel Summary 的数据一致

### 5.2 Launch & Exec Time Trace

![CUDA Kernel Launch & Exec Time Trace](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/720d0d64-dba5-4e26-bf07-78fddb274711/image-019d040c60cb700089314f33b0a72f83-019d0a2a-341f-7f90-a7de-c5d9c2ffa2b6.png)

逐条展示每个 kernel 从 CPU launch 到 GPU 开始执行的延迟。各列含义：

| 列名 | 含义 |
|------|------|
| API Start | CPU 侧 launch API 的开始时间 |
| API Dur | CPU 侧 launch API 耗时 |
| Queue Start / Queue Dur | 在 GPU 命令队列中等待的时间 |
| Kernel Start | GPU 实际开始执行的时间 |
| Kernel Dur | GPU 实际执行耗时 |
| Total Dur | 从 CPU launch 到 GPU 执行完毕的总耗时 |
| GridXYZ / BlockXYZ | Grid 和 Block 的维度配置 |

**实际数据分析**：

| Kernel | API Dur | Queue Dur | Kernel Dur | Total Dur | 说明 |
|--------|---------|-----------|------------|-----------|------|
| `elementwise_kernel` | 49.6µs | - | 51.370µs | 正常 | 小 kernel，执行很快 |
| `unrolled_elementwise_kernel` | 13.4µs | 880ns | 16.399µs | 正常 | 排队时间极短 |
| `ampere_bf16_s16816gemm_..._relu_f2f_stages_64x3_tn` | 26.6µs | **2.224ms** | **218.917µs** | 2.470ms | ⚠️ GEMM 排队等了 2.2ms |
| 大部分 kernel | ~10-13µs | 0.8~2.3ms | 1.3~3.0ms | 2~5ms | Queue Dur >> API Dur |

**关键发现**：
- **所有操作都在 TID 592 上** — 单线程提交所有 kernel
- **Queue Dur（排队时间）远大于 API Dur（提交时间）** — GPU 命令队列中积压了大量任务，GPU 一直在满载运行
- GEMM kernel 排队 2.2ms 才开始执行 — 说明前面有其他 kernel 正在执行
- 这验证了 GPU 利用率高的结论，也说明**当前瓶颈不在 CPU 提交速度上**

---

## 6. CUDA GPU Trace 分析

CUDA GPU Trace 展示了 GPU 上每个操作的逐条执行记录，包括 kernel 和 memory 操作的详细信息。

![CUDA GPU Trace](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/2c3d589a-eb3a-48be-9edb-1dcfd393bc5a/image-019d040c60cb700089314f33b0a72f83-019d0a2a-a73f-7d7b-8a2f-d3f8272b717f.png)

### 6.1 各列含义

| 列名 | 含义 |
|------|------|
| Start | GPU 操作开始时间 |
| Duration | GPU 操作持续时间 |
| CorrId | 关联 ID，可与 CPU 侧的 API 调用对应 |
| GrdX/Y/Z, BlkX/Y/Z | Grid 和 Block 配置（仅 kernel 有） |
| Reg/Trd | 每线程寄存器数 / 每 block 线程数 |
| StcSMem / DynSMem | 静态 / 动态共享内存用量 |
| Bytes | 传输字节数（仅 memcpy/memset 有） |
| Throughput | 传输带宽 |
| SrcMemKd / DstMemKd | 源 / 目标内存类型（Pinned / Device / Pageable） |
| Device | GPU 设备名称 |
| Strm | CUDA stream 编号 |

### 6.2 Memory 操作详细分析

从 Trace 中可以看到开头大量的 **CUDA memcpy Host-to-Device** 操作：

| 传输大小 | 耗时 | Throughput | 内存类型 | 说明 |
|---------|------|------------|----------|------|
| 4B | 1.152µs | 3.31 MiB/s | Pinned → Device | 极小数据（如标量参数），效率极低 |
| 4B | 3.711µs | 1.03 MiB/s | Pinned → Device | 同上 |
| 45.72 KiB | 6.783µs | 6.43 GiB/s | Pinned → Device | 中等数据（如 input_ids），效率合理 |
| 9.38 KiB | 1.536µs | 5.82 GiB/s | Pinned → Device | 合理 |
| 32.09 KiB | 5.472µs | 5.59 GiB/s | Pinned → Device | 合理 |
| **47.81 MiB** | **2.836ms** | **16.44 GiB/s** | Pageable → Device | 🔴 大块传输，可能是首次数据加载 |
| 1.00 KiB | 1.119µs | 876.12 MiB/s | Pinned → Device | 小数据 |
| 96.28 KiB | 9.279ms | 9.90 GiB/s | Pageable → Device | Pageable 内存传输 |

**关键发现**：

1. **大量 4B 的小 memcpy** — Throughput 极低（仅 1~3 MiB/s），这是因为小数据传输的固定开销远大于实际传输时间。建议考虑批量打包这些小传输
2. **47.81 MiB 的大块传输** — Throughput 16.44 GiB/s，接近 PCIe Gen4 x16 的理论带宽（~25 GiB/s），效率不错。但注意使用的是 **Pageable** 内存而非 Pinned，有进一步提升空间
3. **所有操作在 Stream 7 上** — 与前面 Stream 分析一致，计算和 HtoD 传输共用 Default stream
4. **GPU 设备确认**：NVIDIA A100-SXM4-40GB

### 6.3 Kernel 执行详细分析

从 Trace 中可以看到具体的 kernel 执行记录：

| Kernel | Duration | Grid | Block | Reg | StcSMem | DynSMem | 说明 |
|--------|----------|------|-------|-----|---------|---------|------|
| `ampere_bf16_s16816gemm_bf16_128x256_ldg8_...` | **218.917µs** | 8,64,1 | 256,1,1 | 238 | 48.00KiB | 96.00KiB | 大型 GEMM，使用大量共享内存 |
| `elementwise_kernel_with_index` | 2.336µs | 1,1,1 | 64,1,1 | - | 0B | 0B | 极小 kernel |
| `vectorized_elementwise_kernel` | 2.080µs ~ 2.944µs | 1~8,1,1 | 128,1,1 | - | 0B | 0B | 向量化逐元素操作 |
| `unrolled_elementwise_kernel` | 2.432µs ~ 2.751µs | 1,1,1 | 128,1,1 | - | 0B | 0B | 展开的逐元素操作 |

**关键发现**：
- GEMM kernel 使用了 **238 个寄存器/线程** 和 **48KiB 静态 + 96KiB 动态共享内存** — 这是 Tensor Core GEMM 的典型特征，高资源占用确保了高计算吞吐
- 大量小 kernel（Grid = 1,1,1）—— 只用了 1 个 block（1 个 SM），GPU 利用率极低
- Stream 7 上交替出现 memcpy 和 kernel，验证了计算和传输在同一 stream 上串行执行

---

## 7. CUDA GPU Kernel/Grid/Block Summary 分析

在 Kernel Summary 基础上增加了 **Grid 和 Block 配置**信息，用于判断 kernel 的 GPU 并行度是否充分。

![CUDA GPU Kernel/Grid/Block Summary](https://zhiyan-ai-agent-with-1258344702.cos.ap-guangzhou.tencentcos.cn/copilot/5c8105bf-18a2-4ee9-b5db-cc59d74a57af/image-019d040c60cb700089314f33b0a72f83-019d0a29-719c-7059-a08b-27fc34d67990.png)

### 7.1 各 Kernel 的 Grid/Block 配置

| 占比 | 调用次数 | Grid (X,Y,Z) | Block (X,Y,Z) | 总 Block 数 | Kernel | GPU 利用评估 |
|------|----------|--------------|----------------|-------------|--------|-------------|
| 17.3% | 8579 | 2374,1,1 | 128,1,1 | **2374** | `ampere_bf16_s16816gemm_...64x64` | ✅ 优秀：2374 block >> 108 SM |
| 1.3% | 84 | 161,1,16 | 128,1,1 | **2576** | `flash_fwd_splitkv_kernel` | ✅ 优秀 |
| 1.2% | 5060 | **1,1,1** | **1024,1,1** | **1** | `LogSoftMaxForwardEpilogue` | 🔴 极差：仅 1 个 block |
| 1.1% | 364 | 93,13,1 | 256,1,1 | **1209** | `flash_fwd_splitkv_kernel` | ✅ 良好 |
| 1.1% | 84 | 145,1,16 | 128,1,1 | **2320** | `flash_fwd_splitkv_kernel` | ✅ 优秀 |
| 1.0% | 852 | 16,17,1 | 256,1,1 | **272** | `ampere_bf16_s16816gemm_...32x3` | ✅ 良好 |
| 0.9% | 616 | 16,17,1 | 256,1,1 | **272** | `ampere_bf16_s16816gemm_...stages_64x3` | ✅ 良好 |
| 0.8% | 168 | 96,41,1 | 256,1,1 | **3936** | `flash_fwd_splitkv_kernel` | ✅ 优秀 |
| 0.8% | 448 | 96,7,1 | 256,1,1 | **672** | `ampere_bf16_s16816gemm_...128x256` | ✅ 良好 |
| 0.7% | 2876 | **2,1,1** | **1024,1,1** | **2** | `SoftMaxForward` | 🔴 极差：仅 2 个 block |
| 0.6% | 20240 | **1,1,1** | **256,1,1** | **1** | `computeDigitCumSum` | 🔴 极差：仅 1 个 block |

### 7.2 GPU 利用率分析

A100 有 **108 个 SM**，要想 GPU 满载运行，至少需要 108 个 block（实际需要更多，因为 occupancy 和调度开销）。

#### 🔴 低利用率 Kernel（需关注）

| Kernel | Block 数 | SM 利用率 | 调用次数 | 影响 |
|--------|---------|-----------|----------|------|
| `LogSoftMaxForwardEpilogue` | **1** | 0.9% | 5060 | ⚠️ 调用频繁，每次只用 1 个 SM |
| `SoftMaxForward` | **2** | 1.9% | 2876 | ⚠️ 同上 |
| `computeDigitCumSum` | **1** | 0.9% | **20240** | ⚠️ 调用次数最多，GPU 极度浪费 |

这些 kernel 虽然单次执行时间很短（几 µs），但由于**极低的 SM 利用率**，在高频调用下累积的 GPU 浪费不可忽视。
它们是 CUDA Graph 优先覆盖的对象 —— 虽然无法减少单次执行时间，但可以消除 launch overhead。

#### ✅ 高利用率 Kernel

| Kernel | Block 数 | SM 利用率 | 说明 |
|--------|---------|-----------|------|
| GEMM (64x64) | 2374 | **2200%**（多波次） | 每波次填满所有 SM，多波次连续执行 |
| Flash Attention (splitkv) | 2320~3936 | **2100~3600%** | 并行度极高 |
| GEMM (128x256) | 272~1209 | **250~1100%** | 良好 |

### 7.3 Block Size 分析

| Block Size | 线程数 | 使用的 Kernel | 说明 |
|-----------|--------|--------------|------|
| 128,1,1 | 128 | GEMM、Flash Attention | Tensor Core 操作的典型 block size |
| 256,1,1 | 256 | GEMM、Flash Attention、CumSum | 更大 block，适合共享内存密集操作 |
| 1024,1,1 | 1024 | Softmax、LogSoftmax | 最大 block size，但 Grid=1 导致 SM 利用率极低 |

**分析要点**：
- `Softmax` 和 `LogSoftmax` 使用 block=1024 但 grid=1~2，本质上是**将整个操作放在 1~2 个 SM 上串行完成**
- 这可能是因为 Softmax 需要对整个 vocabulary（如 32000+ tokens）做归一化，无法简单并行化
- 如果模型使用了 FlashInfer 或自定义 Triton kernel 的 Softmax 实现，可以改善这一问题

---

## 8. CUDA Graph 原理与分析

### 8.1 传统 CUDA 执行方式的问题

传统方式下，CPU 逐个向 GPU 提交 kernel：

```
CPU                          GPU
 │                            │
 ├── Launch Kernel A -------> │ 执行 Kernel A
 │   (等待 launch 开销 ~5-10µs)│
 ├── Launch Kernel B -------> │ 执行 Kernel B
 │   (等待 launch 开销 ~5-10µs)│
 ├── Launch Kernel C -------> │ 执行 Kernel C
 │   ...                      │
```

对于 LLM 推理，一次 Transformer forward pass 包含 **几百到上千个小 kernel**，每个 kernel 的实际执行时间可能只有几十微秒，
但 launch 开销累计可达 **2.5~5ms**。

### 8.2 CUDA Graph 的解决方案

核心思想：**把一系列 GPU 操作"录制"成一张图（Graph），然后一次性"回放"整张图**。

```
┌─────────── 录制阶段（一次性） ───────────┐
│                                          │
│  CPU: 录制 Kernel A → B → C → ... → Z   │
│  生成一个 Graph 对象                      │
│  实例化为 GraphExec（可执行图）            │
└──────────────────────────────────────────┘

┌─────────── 回放阶段（反复执行） ──────────┐
│                                          │
│  CPU: cudaGraphLaunch(graphExec)         │
│       → 只需一次调用                      │
│                                          │
│  GPU: 自动执行 A → B → C → ... → Z      │
│       （无需 CPU 反复介入）                │
└──────────────────────────────────────────┘
```

### 8.3 性能收益

| 指标 | 无 CUDA Graph | 有 CUDA Graph | 提升 |
|------|--------------|--------------|------|
| 单次 forward 的 CPU launch 开销 | ~500 kernel × 5-10µs = **2.5-5ms** | **1 次 launch ≈ 10-20µs** | **~100x** |
| CPU 利用率 | 高（反复调度） | 低（一次调度） | 显著降低 |
| GPU idle gap | 多（等 CPU 提交下一个 kernel） | 几乎无 | 提高 GPU 利用率 |

**对于 LLM decode 阶段**的典型提升：

```
无 CUDA Graph: 每 token 延迟 ≈ 15ms  (kernel计算 10ms + launch开销 5ms)
有 CUDA Graph: 每 token 延迟 ≈ 10.5ms (kernel计算 10ms + launch开销 0.5ms)
→ 加速比约 30%
```

### 8.4 vLLM 中的 CUDA Graph

**挑战**：每次请求的 batch size 和 sequence length 不同，而 CUDA Graph 要求输入 shape 固定。

**vLLM 的方案**：预录制多个不同 batch size 的 Graph：

```
┌──────────────────────────────────────────┐
│         vLLM CUDA Graph Pool             │
│                                          │
│  Graph[batch=1]   → 1 个请求的 decode    │
│  Graph[batch=2]   → 2 个请求的 decode    │
│  Graph[batch=4]   → 4 个请求的 decode    │
│  Graph[batch=8]   → 8 个请求的 decode    │
│  ...                                     │
│  Graph[batch=256] → 256 个请求的 decode   │
└──────────────────────────────────────────┘
```

运行时 vLLM 会将实际 batch size 向上取整到最近的预录制 size（padding），使用对应的 Graph 执行。

**哪些场景能/不能使用 CUDA Graph**：

| 场景 | 是否使用 | 原因 |
|------|---------|------|
| **Decode 阶段** | ✅ | 每个 token 的计算图结构固定，只是 batch size 不同 |
| **Prefill 阶段** | ❌ | sequence length 变化很大，无法预录制所有 shape |
| **Chunked Prefill** | ❌ | 同上 |

**相关 vLLM 参数**：

```python
--enforce-eager           # 禁用 CUDA Graph（强制 eager 模式，用于调试）
--max-num-seqs 256        # 影响预录制的最大 batch size
--gpu-memory-utilization  # CUDA Graph 会占用额外显存
```

### 8.5 在 nsys 中的体现

从时间线中可以看到 Graphs 和 Kernels 两种执行方式：
- **Graphs（62.2%）**：decode 阶段的 kernel 被打包进 CUDA Graph 一次性执行
- **Kernels（36.7%）**：prefill 阶段的 kernel 仍是逐个 eager launch 的

**实际观察到的 CUDA Graph 预录制数据**：

从 nsys 时间线的 Graphs 展开节点可以看到具体的 Graph 实例：

| Graph ID | GraphExec ID | 占比 | 说明 |
|----------|-------------|------|------|
| Graph 262 | GraphExec 263 | <0.1% | 对应某个 batch size 的 decode 图 |
| Graph 265 | GraphExec 266 | <0.1% | 对应另一个 batch size |
| Graph 268 | GraphExec 269 | <0.1% | ... |
| Graph 271 | GraphExec 272 | <0.1% | ... |
| Graph 274 | GraphExec 275 | <0.1% | ... |
| ... | ... | ... | **114 rows hidden** — 预录制了大量 Graph |

**分析要点**：
- 每个 Graph 对应一个特定的 batch size（或 batch size 区间）
- 单个 Graph 占比 `<0.1%` 是正常的，因为有大量不同的 Graph 被均匀使用
- `114 rows hidden` 说明 vLLM 预录制了 **100+ 个不同 batch size 的 CUDA Graph**
- 每个 Graph 都有一个唯一的 `GraphExec`（Graph ID + 1 = GraphExec ID）

在 Kernel Summary 中，所有列出的 kernel（GEMM、Flash Attention、GELU 等）在 decode 阶段会被打包进 CUDA Graph：

```
┌──── 一次 CUDA Graph Replay ────────────────────────────┐
│                                                         │
│  RMSNorm → QKV_GEMM → RoPE → Attention(splitkv)       │
│  → O_GEMM → RMSNorm → Gate_GEMM → SwiGLU              │
│  → Down_GEMM → ...（重复 N 层）                         │
│                                                         │
│  ← 全部只需 1 次 cudaGraphLaunch                        │
└─────────────────────────────────────────────────────────┘
```

---

## 9. 优化建议总结

### 9.1 量化加速

| 建议 | 条件 | 预期收益 |
|------|------|---------|
| 启用 FP8 量化 (`--quantization fp8`) | GPU 为 H100/H800（当前为 A100 不支持 FP8） | GEMM 接近 2x 加速 |
| 启用 INT8/AWQ 量化 | 精度可接受的场景，A100 支持 INT8 Tensor Core | 减少显存占用，提升吞吐 |

### 9.2 调度优化

| 建议 | 说明 |
|------|------|
| 增加并发请求量 | 减少 GPU 空闲间隙（气泡） |
| 适当增大 `--max-num-seqs` | GEMM kernel 在更大 batch size 下效率更高 |
| 调优 `--max-num-batched-tokens` | 匹配实际负载 |

### 9.3 模型配置优化

| 建议 | 说明 |
|------|------|
| 减少 `--max-model-len` | 如果业务不需要超长上下文，降低它可减少 splitkv attention 耗时 |
| 启用 NVTX 标注 | 获取 prefill/decode/schedule 各阶段的精确耗时 |

### 9.4 Kernel 并行度优化

| 建议 | 说明 |
|------|------|
| 优化 Softmax/LogSoftmax 实现 | 当前仅使用 1~2 个 block，GPU SM 利用率极低（<2%）。可考虑使用 FlashInfer 或自定义 Triton kernel |
| 合并小 kernel | `computeDigitCumSum` 等 Grid=1 的 kernel 被调用 2 万+ 次，建议通过 kernel fusion 减少调用次数 |
| 避免小数据 memcpy | 大量 4B 的 HtoD 传输效率极低（~1 MiB/s），建议批量打包 |
| 使用 Pinned Memory | GPU Trace 显示部分大块传输使用 Pageable 内存，改为 Pinned 可提升带宽 |

### 9.5 分析那 37% 未被 CUDA Graph 捕获的 Kernel

可能来源：
- Prefill 阶段（变长输入无法用 CUDA Graph）
- 视觉编码器部分（多模态模型的图像/视频处理）

---

## 10. 常用操作备忘

### 10.1 nsys 采集命令

```bash
# 基础采集
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o report <command>

# 带时间限制的采集
nsys profile --trace=cuda,nvtx --duration=30 -o report <command>

# 低开销采集（减少 Profiler Overhead）
nsys profile --trace=cuda --backtrace=none --duration=10 -o report <command>

# 采样模式（最低开销）
nsys profile --sample=cpu --trace=cuda -o report <command>
```

### 10.2 命令行导出统计

```bash
# 导出 Kernel Summary
nsys stats -r cuda_gpu_kern_sum report.nsys-rep

# 导出 CUDA API Summary
nsys stats -r cuda_api_sum report.nsys-rep

# 导出 Kernel Launch & Exec Time Summary
nsys stats -r cuda_kern_exec_sum report.nsys-rep

# 导出 Kernel Launch & Exec Time Trace（逐条记录）
nsys stats -r cuda_kern_exec_trace report.nsys-rep

# 导出 GPU Trace（逐条 kernel + memcpy 记录）
nsys stats -r cuda_gpu_trace report.nsys-rep

# 导出 Kernel/Grid/Block Summary
nsys stats -r cuda_gpu_kern_gb_sum report.nsys-rep
```

### 10.3 推荐查看顺序

```
1. CUDA GPU Kernel Summary        ← 哪些 kernel 最耗时？
     ↓
2. CUDA API Summary               ← CPU 侧花在哪些 API 上？同步等待占多少？
     ↓
3. Kernel Launch & Exec Time Sum  ← kernel 排队等待了多久？CPU 提交是否跟得上？
     ↓
4. CUDA GPU Kernel/Grid/Block Sum ← kernel 的 GPU 并行度是否充分？
     ↓
5. CUDA GPU Trace                 ← 逐条排查：哪些 memcpy 效率低？哪个 kernel 实例异常？
     ↓
6. 如有异常 → 对应的 Trace 版本逐条排查
```

### 10.4 GUI 操作要点

1. 打开 `.nsys-rep` 文件后，在左下角 **Stats System View** 列表中单击对应报告项
2. **CUDA GPU Kernel Summary** — 查看 kernel 耗时排行
3. **CUDA GPU Trace** — 查看每个 kernel 的执行时间线
4. **CUDA Kernel Launch & Exec Time Summary** — 查看 launch 延迟
5. 如果某个报告为空，检查采集命令是否包含 `--trace=cuda` 参数
