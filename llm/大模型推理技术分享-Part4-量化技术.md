# 大模型推理技术分享 - Part 4: 量化技术基础

> **受众**: AI平台开发工程师
> **目标**: 理解从CV到大模型的量化技术演进

---

## 目录

1. [量化基础概念](#1-量化基础概念)
2. [传统CV量化](#2-传统cv量化)
3. [大模型量化特点](#3-大模型量化特点)
4. [主流量化方法](#4-主流量化方法)
5. [量化实践](#5-量化实践)

---

## 1. 量化基础概念

### 1.1 什么是量化？

**定义**：将高精度浮点数（FP32/FP16）映射到低精度整数（INT8/INT4）的过程。

**目标**：
- 减少模型大小（4-8x）
- 减少内存带宽（4-8x）
- 加速推理（2-4x）
- 降低功耗

**核心公式**：
```
量化：x_int = round(x_float / scale) + zero_point
反量化：x_float = (x_int - zero_point) × scale
```

### 1.2 数据类型对比

| 类型 | 位数 | 范围 | 精度 | 用途 |
|------|------|------|------|------|
| **FP32** | 32 | ±3.4×10³⁸ | 7位有效数字 | 训练（标准） |
| **FP16** | 16 | ±65504 | 3位有效数字 | 训练/推理 |
| **BF16** | 16 | ±3.4×10³⁸ | 2位有效数字 | 训练（稳定） |
| **FP8** | 8 | ±57344 | 2位有效数字 | 推理（H100） |
| **INT8** | 8 | -128~127 | 整数 | 推理（通用） |
| **INT4** | 4 | -8~7 | 整数 | 推理（极致压缩） |

**浮点格式对比**：
```
FP32: 1 sign + 8 exponent + 23 mantissa
FP16: 1 sign + 5 exponent + 10 mantissa
BF16: 1 sign + 8 exponent + 7 mantissa  (与FP32范围相同)
FP8:  1 sign + 4 exponent + 3 mantissa
```

### 1.3 量化的挑战

**精度损失**：
```
FP32: 3.141592653589793
FP16: 3.140625
INT8: 3 (scale=1)
INT4: 3 (scale=1)
```

**异常值（Outliers）**：
```
正常分布: [-1, 1]
异常值: [-100, 100]

如果按异常值设置scale，正常值精度损失严重
如果按正常值设置scale，异常值会溢出
```

---

## 2. 传统CV量化

### 2.1 对称量化 vs 非对称量化

#### 对称量化（Symmetric Quantization）
```
公式：x_int = round(x_float / scale)
     scale = max(|x_float|) / 127

特点：
- zero_point = 0
- 范围对称：[-127, 127]
- 实现简单，硬件友好
```

**示例**：
```python
def symmetric_quantize(x, bits=8):
    # 计算scale
    max_val = torch.max(torch.abs(x))
    scale = max_val / (2 ** (bits - 1) - 1)
    
    # 量化
    x_int = torch.round(x / scale)
    x_int = torch.clamp(x_int, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    
    return x_int, scale

def symmetric_dequantize(x_int, scale):
    return x_int * scale
```

#### 非对称量化（Asymmetric Quantization）
```
公式：x_int = round(x_float / scale) + zero_point
     scale = (max(x_float) - min(x_float)) / 255
     zero_point = round(-min(x_float) / scale)

特点：
- zero_point ≠ 0
- 范围非对称：[0, 255] 或 [-128, 127]
- 精度更高，但计算复杂
```

**示例**：
```python
def asymmetric_quantize(x, bits=8):
    # 计算scale和zero_point
    min_val = torch.min(x)
    max_val = torch.max(x)
    
    qmin = 0
    qmax = 2 ** bits - 1
    
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)
    
    # 量化
    x_int = torch.round(x / scale) + zero_point
    x_int = torch.clamp(x_int, qmin, qmax)
    
    return x_int, scale, zero_point

def asymmetric_dequantize(x_int, scale, zero_point):
    return (x_int - zero_point) * scale
```

**对比**：
```
数据: [-5, -3, 0, 2, 10]

对称量化:
scale = 10 / 127 = 0.0787
量化后: [-64, -38, 0, 25, 127]
反量化: [-5.04, -2.99, 0, 1.97, 10.0]

非对称量化:
scale = (10 - (-5)) / 255 = 0.0588
zero_point = 0 - (-5) / 0.0588 = 85
量化后: [0, 34, 85, 119, 255]
反量化: [-5.0, -3.0, 0, 2.0, 10.0]

结论：非对称量化精度更高
```

### 2.2 Per-Tensor vs Per-Channel 量化

#### Per-Tensor 量化
```
整个Tensor使用同一个scale

优点：实现简单，内存开销小
缺点：精度较低（不同channel分布差异大）
```

**示例**：
```python
def per_tensor_quantize(x):
    # x: [out_channels, in_channels, H, W]
    scale = torch.max(torch.abs(x)) / 127
    x_int = torch.round(x / scale)
    return x_int, scale
```

#### Per-Channel 量化
```
每个输出channel使用独立的scale

优点：精度更高
缺点：内存开销大（需要存储多个scale）
```

**示例**：
```python
def per_channel_quantize(x):
    # x: [out_channels, in_channels, H, W]
    out_channels = x.shape[0]
    
    scales = []
    x_int = torch.zeros_like(x, dtype=torch.int8)
    
    for i in range(out_channels):
        scale = torch.max(torch.abs(x[i])) / 127
        x_int[i] = torch.round(x[i] / scale)
        scales.append(scale)
    
    return x_int, torch.tensor(scales)
```

**对比**：
```
权重分布:
Channel 0: [-1, 1]
Channel 1: [-10, 10]

Per-Tensor:
scale = 10 / 127 = 0.0787
Channel 0 量化后: [-13, 13]  (精度损失大)
Channel 1 量化后: [-127, 127]

Per-Channel:
Channel 0: scale = 1 / 127 = 0.0079, 量化后: [-127, 127]
Channel 1: scale = 10 / 127 = 0.0787, 量化后: [-127, 127]

结论：Per-Channel精度更高
```

### 2.3 校准（Calibration）

**目的**：确定最优的scale和zero_point

**方法**：

#### 1. Min-Max 校准
```python
def min_max_calibration(x):
    """最简单的方法：使用最小值和最大值"""
    min_val = torch.min(x)
    max_val = torch.max(x)
    
    scale = (max_val - min_val) / 255
    zero_point = -torch.round(min_val / scale)
    
    return scale, zero_point
```

**问题**：对异常值敏感

#### 2. Percentile 校准
```python
def percentile_calibration(x, percentile=99.99):
    """使用百分位数，忽略异常值"""
    min_val = torch.quantile(x, (100 - percentile) / 100)
    max_val = torch.quantile(x, percentile / 100)
    
    scale = (max_val - min_val) / 255
    zero_point = -torch.round(min_val / scale)
    
    return scale, zero_point
```

**优点**：对异常值鲁棒

#### 3. KL散度校准（TensorRT）
```python
def kl_divergence_calibration(x, num_bins=2048):
    """最小化量化前后的KL散度"""
    # 1. 构造直方图
    hist, bin_edges = torch.histogram(x, bins=num_bins)
    
    # 2. 搜索最优threshold
    best_threshold = None
    min_kl_div = float('inf')
    
    for threshold in candidate_thresholds:
        # 量化
        x_quant = quantize(x, threshold)
        
        # 计算KL散度
        kl_div = compute_kl_divergence(x, x_quant)
        
        if kl_div < min_kl_div:
            min_kl_div = kl_div
            best_threshold = threshold
    
    scale = best_threshold / 127
    return scale
```

**优点**：精度最高，但计算复杂

---

## 3. 大模型量化特点

### 3.1 大模型 vs CV模型

| 特性 | CV模型 | 大模型 |
|------|--------|--------|
| **参数量** | 百万-千万 | 十亿-千亿 |
| **激活值分布** | 相对均匀 | 存在严重异常值 |
| **量化粒度** | Per-Channel | Per-Token, Per-Group |
| **量化难度** | 较低 | 较高 |
| **主要瓶颈** | 计算 | 内存带宽 |

### 3.2 大模型的异常值问题

**现象**：
```
正常激活值: [-1, 1]
异常值: [-100, 100]

异常值占比: <0.1%
但影响巨大: 决定了scale的范围
```

**示例**（Llama-2-7B）：
```python
# 激活值分布
activations = model.forward(input)
print(f"Mean: {activations.mean()}")
print(f"Std: {activations.std()}")
print(f"Max: {activations.max()}")
print(f"Min: {activations.min()}")

# 输出:
# Mean: 0.02
# Std: 0.8
# Max: 127.5  ← 异常值！
# Min: -98.3  ← 异常值！
```

**影响**：
```
如果按异常值设置scale:
scale = 127.5 / 127 = 1.0

正常值量化:
x = 0.8 → x_int = round(0.8 / 1.0) = 1
反量化: 1 × 1.0 = 1.0
误差: |1.0 - 0.8| / 0.8 = 25%  ← 精度损失严重！
```

### 3.3 解决方案

#### 1. SmoothQuant
**核心思想**：将激活值的异常值转移到权重

```
原始: Y = X @ W
     X: 激活值（有异常值）
     W: 权重（分布均匀）

SmoothQuant: Y = (X / s) @ (W × s)
            X/s: 激活值（异常值被平滑）
            W×s: 权重（吸收了scale）
```

**实现**：
```python
def smooth_quant(X, W):
    # 1. 计算平滑因子
    s = torch.max(torch.abs(X), dim=0) ** alpha / \
        torch.max(torch.abs(W), dim=1) ** (1 - alpha)
    
    # 2. 平滑激活值
    X_smooth = X / s
    
    # 3. 调整权重
    W_smooth = W * s.unsqueeze(1)
    
    # 4. 量化
    X_int, scale_x = quantize(X_smooth)
    W_int, scale_w = quantize(W_smooth)
    
    return X_int, W_int, scale_x, scale_w
```

**效果**：
```
原始:
X: [-100, 100], W: [-1, 1]
量化后精度损失: 30%

SmoothQuant:
X_smooth: [-10, 10], W_smooth: [-10, 10]
量化后精度损失: 5%
```

#### 2. AWQ (Activation-aware Weight Quantization)
**核心思想**：保护重要的权重通道

```
观察：少数权重通道对应大的激活值，对输出影响大

策略：
1. 识别重要通道（激活值大的通道）
2. 对重要通道使用更高精度或更小的scale
3. 对不重要通道使用更低精度
```

**实现**：
```python
def awq_quantize(X, W):
    # 1. 计算每个通道的重要性
    importance = torch.mean(torch.abs(X), dim=0)
    
    # 2. 对重要通道使用更小的scale
    scales = []
    for i in range(W.shape[0]):
        if importance[i] > threshold:
            # 重要通道：使用per-channel scale
            scale = torch.max(torch.abs(W[i])) / 127
        else:
            # 不重要通道：使用per-tensor scale
            scale = torch.max(torch.abs(W)) / 127
        scales.append(scale)
    
    # 3. 量化
    W_int = torch.round(W / torch.tensor(scales).unsqueeze(1))
    
    return W_int, scales
```

#### 3. GPTQ (Generative Pre-trained Transformer Quantization)
**核心思想**：逐层量化，最小化重构误差

```
目标：min ||WX - W_quantX||²

方法：
1. 固定输入X
2. 逐层量化权重W
3. 使用Hessian矩阵指导量化顺序
```

**实现**：
```python
def gptq_quantize(W, X):
    # 1. 计算Hessian矩阵
    H = 2 * X.T @ X
    
    # 2. 逐列量化
    W_quant = torch.zeros_like(W)
    for i in range(W.shape[1]):
        # 量化第i列
        w_int, scale = quantize(W[:, i])
        W_quant[:, i] = dequantize(w_int, scale)
        
        # 补偿误差（关键！）
        error = W[:, i] - W_quant[:, i]
        W[:, i+1:] -= (error.unsqueeze(1) @ H[i, i+1:].unsqueeze(0)) / H[i, i]
    
    return W_quant
```

---

## 4. 主流量化方法

### 4.1 PTQ (Post-Training Quantization)

**定义**：训练后量化，不需要重新训练

**流程**：
```
1. 加载预训练模型（FP16）
2. 使用校准数据集确定scale
3. 量化权重和激活值
4. 评估精度
```

**优点**：
- 简单快速
- 不需要训练数据
- 不需要GPU资源

**缺点**：
- 精度损失较大（1-5%）
- 对异常值敏感

**代码示例**：
```python
from transformers import AutoModelForCausalLM
import torch

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 校准
calibration_data = load_calibration_data()
with torch.no_grad():
    for batch in calibration_data:
        _ = model(batch)

# 3. 量化
from torch.quantization import quantize_dynamic
model_int8 = quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# 4. 保存
model_int8.save_pretrained("llama-2-7b-int8")
```

### 4.2 QAT (Quantization-Aware Training)

**定义**：训练时模拟量化，让模型适应量化误差

**流程**：
```
1. 插入伪量化节点（Fake Quantization）
2. 前向传播：量化 → 反量化
3. 反向传播：使用STE（Straight-Through Estimator）
4. 训练完成后，移除伪量化节点
```

**伪量化**：
```python
class FakeQuantize(nn.Module):
    def forward(self, x):
        # 量化
        x_int = torch.round(x / self.scale)
        x_int = torch.clamp(x_int, -128, 127)
        
        # 反量化
        x_quant = x_int * self.scale
        
        # STE: 反向传播时，梯度直通
        return x_quant
```

**优点**：
- 精度损失小（<1%）
- 对异常值鲁棒

**缺点**：
- 需要重新训练（成本高）
- 需要训练数据

### 4.3 混合精度量化

**核心思想**：不同层使用不同精度

```
敏感层（Attention）: FP16
不敏感层（FFN）: INT8
极不敏感层: INT4
```

**示例**：
```python
class MixedPrecisionModel(nn.Module):
    def __init__(self, model):
        self.model = model
        
        # 标记敏感层
        self.sensitive_layers = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ]
    
    def quantize(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if any(s in name for s in self.sensitive_layers):
                    # 敏感层：FP16
                    module.weight.data = module.weight.data.half()
                else:
                    # 不敏感层：INT8
                    module.weight.data = quantize_int8(module.weight.data)
```

### 4.4 主流方法对比

| 方法 | 精度损失 | 速度 | 内存 | 成本 | 适用场景 |
|------|---------|------|------|------|---------|
| **PTQ** | 2-5% | 快 | 低 | 低 | 快速部署 |
| **QAT** | <1% | 慢 | 高 | 高 | 高精度要求 |
| **SmoothQuant** | 1-2% | 快 | 低 | 低 | 通用 |
| **AWQ** | 1-2% | 快 | 低 | 中 | 推荐 |
| **GPTQ** | 1-3% | 中 | 低 | 中 | 极致压缩 |

---

## 5. 量化实践

### 5.1 使用 vLLM + AWQ

**步骤1：量化模型**
```bash
# 安装autoawq
pip install autoawq

# 量化脚本
python -m awq.entry --model_path meta-llama/Llama-2-7b-hf \
    --w_bit 4 \
    --q_group_size 128 \
    --output_path llama-2-7b-awq
```

**步骤2：使用vLLM加载**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model llama-2-7b-awq \
    --quantization awq \
    --dtype half \
    --max-model-len 4096
```

**性能对比**：
```
FP16:
- 内存: 14 GB
- 吞吐量: 2800 tokens/s

AWQ (INT4):
- 内存: 4 GB  (3.5x 减少)
- 吞吐量: 3200 tokens/s  (1.14x 提升)
- 精度损失: <2%
```

### 5.2 使用 TensorRT-LLM + FP8

**步骤1：量化模型**
```python
import tensorrt_llm
from tensorrt_llm.quantization import quantize_fp8

# 加载模型
model = tensorrt_llm.LLaMAForCausalLM.from_hugging_face(
    "meta-llama/Llama-2-7b-hf"
)

# FP8量化
model_fp8 = quantize_fp8(
    model,
    calibration_dataset=calibration_data,
)

# 构建Engine
engine = tensorrt_llm.build(
    model_fp8,
    max_batch_size=32,
    max_input_len=2048,
    max_output_len=512,
)

# 保存
engine.save("llama-2-7b-fp8.engine")
```

**步骤2：推理**
```python
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("llama-2-7b-fp8.engine")

outputs = runner.generate(
    input_ids=input_ids,
    max_new_tokens=100,
)
```

**性能对比**（H100）：
```
FP16:
- 吞吐量: 3500 tokens/s

FP8:
- 吞吐量: 6800 tokens/s  (1.94x 提升)
- 精度损失: <1%
```

### 5.3 量化评估

**指标**：
```python
# 1. 困惑度（Perplexity）
def evaluate_perplexity(model, dataset):
    total_loss = 0
    for batch in dataset:
        outputs = model(batch)
        loss = F.cross_entropy(outputs.logits, batch.labels)
        total_loss += loss.item()
    
    perplexity = torch.exp(torch.tensor(total_loss / len(dataset)))
    return perplexity

# 2. 准确率（Accuracy）
def evaluate_accuracy(model, dataset):
    correct = 0
    total = 0
    for batch in dataset:
        outputs = model(batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch.labels).sum().item()
        total += batch.labels.numel()
    
    accuracy = correct / total
    return accuracy

# 3. 下游任务性能
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "winogrande", "arc_easy", "arc_challenge"],
)
```

**示例结果**：
```
Llama-2-7B:
- FP16: Perplexity=5.68, Accuracy=76.2%
- INT8 (SmoothQuant): Perplexity=5.72, Accuracy=75.8%
- INT4 (AWQ): Perplexity=5.81, Accuracy=74.9%
- INT4 (GPTQ): Perplexity=5.89, Accuracy=74.2%
```

### 5.4 最佳实践

**选择量化方法**：
```
1. 快速部署：PTQ (SmoothQuant)
2. 高精度要求：QAT 或 AWQ
3. 极致压缩：GPTQ (INT4)
4. H100 GPU：FP8
```

**量化粒度**：
```
1. 权重：Per-Channel 或 Per-Group
2. 激活值：Per-Token
3. KV Cache：Per-Token (重要！)
```

**校准数据**：
```
1. 数量：512-1024 samples
2. 来源：与推理数据分布一致
3. 长度：覆盖不同长度（512, 1024, 2048）
```

**监控指标**：
```
1. 精度：Perplexity, Accuracy
2. 性能：吞吐量, 延迟
3. 内存：模型大小, KV Cache大小
```

---

## 总结

### 核心要点

1. **量化基础**：
   - 对称 vs 非对称
   - Per-Tensor vs Per-Channel
   - 校准方法：Min-Max, Percentile, KL散度

2. **大模型量化特点**：
   - 异常值问题严重
   - 需要特殊处理：SmoothQuant, AWQ, GPTQ

3. **量化方法**：
   - PTQ：快速，精度损失2-5%
   - QAT：慢，精度损失<1%
   - 混合精度：平衡精度和性能

4. **实践建议**：
   - 通用场景：AWQ (INT4)
   - H100：FP8
   - 极致压缩：GPTQ (INT4)
   - 评估：Perplexity + 下游任务

### 量化效果总结

| 方法 | 压缩比 | 加速比 | 精度损失 | 推荐度 |
|------|--------|--------|---------|--------|
| **FP16** | 1x | 1x | 0% | ⭐⭐⭐ |
| **INT8 (SmoothQuant)** | 2x | 1.5x | 1-2% | ⭐⭐⭐⭐ |
| **INT4 (AWQ)** | 4x | 2x | 1-2% | ⭐⭐⭐⭐⭐ |
| **INT4 (GPTQ)** | 4x | 2x | 2-3% | ⭐⭐⭐⭐ |
| **FP8 (H100)** | 2x | 2x | <1% | ⭐⭐⭐⭐⭐ |

---

## 完整分享总结

### 四大模块回顾

1. **基础概念**：KV Cache, Prefill/Decode, FlashAttention, Continuous Batching, 并行策略, 推测解码
2. **GPU与算子**：GPU架构, 内存层次, CUDA编程, 关键算子优化, 性能分析
3. **推理框架**：vLLM, SGLang, TensorRT-LLM对比, vLLM核心设计, 生产实践
4. **量化技术**：量化基础, CV vs 大模型, 主流方法, 实践指南

### 技术选型建议

**场景1：通用推理服务**
```
框架：vLLM
量化：AWQ (INT4)
并行：TP=8 (单节点)
配置：--enable-prefix-caching
```

**场景2：Few-shot / 对话**
```
框架：SGLang
量化：AWQ (INT4)
并行：TP=4
特性：RadixAttention自动前缀缓存
```

**场景3：极致性能（H100）**
```
框架：TensorRT-LLM
量化：FP8
并行：TP=8
特性：深度编译优化
```

**场景4：超大模型（70B+）**
```
框架：vLLM
量化：AWQ (INT4)
并行：TP=8, PP=4 (跨节点)
配置：--max-model-len 4096
```

### 关键性能指标

**吞吐量优化**：
- 增加 batch size
- 启用 prefix caching
- 使用量化（INT4/FP8）
- Continuous Batching

**延迟优化**：
- 减少 batch size
- Chunked Prefill
- CUDA Graph
- Speculative Decoding

**内存优化**：
- PagedAttention / RadixAttention
- 量化（INT4）
- 调整 max_model_len
- KV Cache 管理

---

**分享完毕！祝大家在大模型推理优化的道路上越走越远！** 🚀
