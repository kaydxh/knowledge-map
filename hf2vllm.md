# HuggingFace 到 vLLM 实现转换完整指南

## 📋 目录
1. [核心对应关系](#核心对应关系)
2. [逐步转换流程](#逐步转换流程)
3. [详细代码映射](#详细代码映射)
4. [常见转换模式](#常见转换模式)
5. [完整转换模板](#完整转换模板)

---

## 1. 核心对应关系

### 1.1 整体架构映射

```
HuggingFace 实现                          vLLM 实现
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────┐    ┌──────────────────────────────┐
│ class Qwen2ForRewardModel       │    │ class Qwen2ForRewardModel    │
│   (PreTrainedModel)             │ → │   (nn.Module)                │
│                                 │    │                              │
│ - __init__(config)              │    │ - __init__(config,           │
│                                 │    │     cache_config,            │
│                                 │    │     quant_config,            │
│                                 │    │     lora_config)             │
│                                 │    │                              │
│ - self.model = Qwen2Model()     │    │ - self.model = Qwen2Model()  │
│                                 │    │                              │
│ - self.score = nn.Sequential(   │    │ - self.score = nn.Sequential(│
│     nn.Linear(),                │    │     ColumnParallelLinear(),  │
│     nn.ReLU(),                  │    │     ReLU(),  ← 自定义         │
│     nn.Linear()                 │    │     RowParallelLinear()      │
│   )                             │    │   )                          │
│                                 │    │                              │
│ - forward(                      │    │ - forward(                   │
│     input_ids,                  │    │     input_ids,               │
│     attention_mask,             │    │     positions,               │
│     position_ids,               │    │     kv_caches,               │
│     past_key_values,            │    │     attn_metadata,           │
│     labels,                     │    │     intermediate_tensors     │
│     ...                         │    │   )                          │
│   )                             │    │                              │
│   → 返回 SequenceClassifier     │    │   → 返回 torch.Tensor        │
│      OutputWithPast             │    │                              │
│                                 │    │ - pooler(                    │
│ - 内部处理pooling                │    │     hidden_states,           │
│   (在forward中)                  │    │     pooling_metadata         │
│                                 │    │   )                          │
│                                 │    │                              │
│ - 自动权重加载                   │    │ - load_weights(weights)      │
│   (from_pretrained)             │    │   ← 手动实现                  │
└─────────────────────────────────┘    └──────────────────────────────┘
```

### 1.2 逐层对应关系

| 组件 | HuggingFace | vLLM | 关键差异 |
|------|-------------|------|---------|
| **基类** | `PreTrainedModel` | `nn.Module` | vLLM不需要HF的模型管理功能 |
| **初始化参数** | `config` | `config, cache_config, quant_config, lora_config` | vLLM需要推理相关配置 |
| **线性层** | `nn.Linear` | `ColumnParallelLinear` / `RowParallelLinear` | vLLM支持tensor并行 |
| **激活函数** | `nn.ReLU()` | `ReLU()` (自定义) | vLLM需要处理元组输出 |
| **前向参数** | 训练友好(labels, return_dict) | 推理优化(kv_caches, attn_metadata) | 不同使用场景 |
| **输出格式** | 结构化对象 | 原始张量 | vLLM追求性能 |
| **池化** | 内嵌在forward | 独立的pooler方法 | vLLM解耦逻辑 |
| **权重加载** | 自动化 | 手动实现 | vLLM需要处理并行化 |

---

## 2. 逐步转换流程

### 步骤 1: 分析 HuggingFace 实现结构

```python
# 1. 识别模型的核心组件
class Qwen2ForRewardModel(PreTrainedModel):
    def __init__(self, config):
        # 找出所有子模块
        self.model = Qwen2Model(config)           # ← 骨干网络
        self.score = nn.Sequential(...)           # ← 任务特定头
    
    def forward(self, ...):
        # 找出计算流程
        hidden_states = self.model(...)           # ← 特征提取
        logits = self.score(hidden_states)        # ← 任务计算
        pooled_logits = logits[..., -1, :]       # ← 池化逻辑
        return SequenceClassifierOutput(...)      # ← 输出格式
```

**分析清单**:
- ✅ 骨干网络: `self.model`
- ✅ 任务头: `self.score`
- ✅ 计算流程: model → score → pooling
- ✅ 输入参数: input_ids, attention_mask, position_ids...
- ✅ 输出格式: SequenceClassifierOutputWithPast

### 步骤 2: 创建 vLLM 基础框架

```python
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, RowParallelLinear
)
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.qwen2 import Qwen2Model

class Qwen2ForRewardModel(nn.Module):
    # 1. 添加vLLM特定的类属性
    packed_modules_mapping = {...}      # ← 参数堆叠映射
    supported_lora_modules = [...]      # ← LoRA支持
    
    def __init__(
        self,
        config,
        cache_config=None,              # ← vLLM专用
        quant_config=None,              # ← vLLM专用
        lora_config=None,               # ← vLLM专用
    ):
        super().__init__()
        # 2. 初始化配置
        self.config = config
        self.quant_config = quant_config
        
        # 3. 初始化骨干网络
        self.model = Qwen2Model(config, cache_config, quant_config)
        
        # 4. 初始化任务头 (待转换)
        # 5. 初始化pooler (新增)
```

### 步骤 3: 转换线性层为并行层

```python
# HuggingFace → vLLM 转换规则

# 规则1: 输出层使用 ColumnParallelLinear
nn.Linear(in_features, out_features, bias=True)
↓
ColumnParallelLinear(
    in_features,
    out_features,
    bias=True,                          # 保持原有bias设置
    quant_config=quant_config,          # 新增: 量化配置
)

# 规则2: 最终投影层使用 RowParallelLinear
nn.Linear(in_features, out_features, bias=False)
↓
RowParallelLinear(
    in_features,
    out_features,
    bias=False,
    quant_config=quant_config,
)

# 规则3: 中间的ReLU需要自定义包装
nn.ReLU()
↓
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
    
    def forward(self, input):
        input, _ = input  # ← 解包ColumnParallelLinear的元组输出
        return self.activation(input)
```

**实际转换例子**:

```python
# ============ HuggingFace 版本 ============
self.score = nn.Sequential(
    nn.Linear(config.hidden_size, config.hidden_size),
    nn.ReLU(),
    nn.Linear(config.hidden_size, 1)
)

# ============ vLLM 版本 ============
self.score = nn.Sequential(
    ColumnParallelLinear(                    # 第1层: 列并行
        config.hidden_size,
        config.hidden_size,
        quant_config=quant_config,
    ),
    ReLU(),                                   # 激活: 自定义处理元组
    RowParallelLinear(                        # 第2层: 行并行
        config.hidden_size,
        1,
        quant_config=quant_config,
    ),
)
```

### 步骤 4: 转换 forward 方法

```python
# ============ HuggingFace 版本 ============
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,          # ← 训练用
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, SequenceClassifierOutputWithPast]:
    # 1. 调用骨干网络
    transformer_outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        ...
    )
    hidden_states = transformer_outputs[0]
    
    # 2. 计算logits
    logits = self.score(hidden_states)
    
    # 3. 池化 (提取最后一个token)
    sequence_lengths = ... # 计算有效长度
    pooled_logits = logits[torch.arange(batch_size), sequence_lengths]
    
    # 4. 计算损失 (如果有labels)
    loss = None
    if labels is not None:
        loss_fct = MSELoss()
        loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
    
    # 5. 返回结构化输出
    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=transformer_outputs.past_key_values,
        ...
    )

# ============ vLLM 版本 ============
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,                             # ← 简化的位置信息
    kv_caches: List[torch.Tensor],                      # ← vLLM的KV缓存
    attn_metadata: AttentionMetadata,                   # ← 注意力元数据
    intermediate_tensors: Optional[IntermediateTensors] = None,  # ← 流水线并行
) -> torch.Tensor:                                      # ← 简单返回张量
    # 1. 调用骨干网络
    hidden_states = self.model(
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors
    )
    
    # 2. 计算logits
    logits, _ = self.score(hidden_states)  # ← 注意解包元组
    
    # 3. 直接返回 (池化在单独的pooler方法中)
    return logits

# 新增: 独立的pooler方法
def pooler(
    self,
    hidden_states: torch.Tensor,
    pooling_metadata: PoolingMetadata,
) -> Optional[PoolerOutput]:
    return self._pooler(hidden_states, pooling_metadata)
```

**转换要点**:
- ❌ 移除训练相关参数: `labels`, `return_dict`
- ❌ 移除输出控制参数: `output_attentions`, `output_hidden_states`
- ✅ 使用vLLM参数: `positions`, `kv_caches`, `attn_metadata`
- ✅ 简化返回值: 直接返回张量，不返回损失和元数据
- ✅ 池化解耦: 从forward移到独立的pooler方法

### 步骤 5: 实现权重加载

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    """
    核心逻辑:
    1. 定义参数映射关系
    2. 遍历所有权重
    3. 处理参数名映射
    4. 调用权重加载器
    """
    
    # 1. 定义堆叠参数映射
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    
    # 2. 获取模型参数字典
    params_dict = dict(self.named_parameters(remove_duplicate=False))
    
    # 3. 遍历权重
    for name, loaded_weight in weights:
        # 3.1 跳过不需要的权重
        if name == "lm_head.weight":
            continue
        if "rotary_emb.inv_freq" in name:
            continue
        
        # 3.2 处理堆叠参数
        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            if weight_name not in name:
                continue
            
            # 替换参数名
            name = name.replace(weight_name, param_name)
            
            # 跳过不存在的bias
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            # 加载权重
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            # 3.3 处理常规参数
            if name.endswith(".bias") and name not in params_dict:
                continue
            
            name = maybe_remap_kv_scale_name(name, params_dict)
            if name is None:
                continue
            
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
```

### 步骤 6: 添加必要的类属性

```python
class Qwen2ForRewardModel(nn.Module):
    # 1. 参数堆叠映射 (用于权重加载)
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    # 2. LoRA支持的模块
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    
    # 3. Embedding相关 (通常为空)
    embedding_modules = {}
    embedding_padding_modules = []
```

---

## 3. 详细代码映射表

### 3.1 初始化方法映射

| 步骤 | HuggingFace | vLLM | 说明 |
|------|-------------|------|------|
| 1 | `super().__init__(config)` | `super().__init__()` | vLLM不继承PreTrainedModel |
| 2 | `self.config = config` | `self.config = config`<br>`self.quant_config = quant_config`<br>`self.lora_config = lora_config` | vLLM需要额外配置 |
| 3 | `self.model = Qwen2Model(config)` | `self.model = Qwen2Model(config, cache_config, quant_config)` | vLLM传入推理配置 |
| 4 | `self.score = nn.Sequential(`<br>`  nn.Linear(...),`<br>`  nn.ReLU(),`<br>`  nn.Linear(...)`<br>`)` | `self.score = nn.Sequential(`<br>`  ColumnParallelLinear(...),`<br>`  ReLU(),`<br>`  RowParallelLinear(...)`<br>`)` | 使用并行层 |
| 5 | 无 | `self._pooler = Pooler(...)` | vLLM新增pooler |
| 6 | `self.post_init()` | 无 | vLLM不需要 |

### 3.2 Forward方法映射

| 部分 | HuggingFace | vLLM | 转换说明 |
|------|-------------|------|---------|
| **参数** | `input_ids, attention_mask, position_ids, past_key_values, labels, use_cache, output_attentions, output_hidden_states, return_dict` | `input_ids, positions, kv_caches, attn_metadata, intermediate_tensors` | 移除训练/调试参数，使用vLLM推理参数 |
| **骨干调用** | `transformer_outputs = self.model(...)`<br>`hidden_states = transformer_outputs[0]` | `hidden_states = self.model(...)` | vLLM直接返回张量 |
| **Score计算** | `logits = self.score(hidden_states)` | `logits, _ = self.score(hidden_states)` | vLLM需要解包元组 |
| **池化** | 在forward内部:<br>`sequence_lengths = ...`<br>`pooled = logits[..., sequence_lengths]` | 移到pooler方法:<br>`def pooler(self, ...)` | 逻辑解耦 |
| **损失计算** | `if labels is not None:`<br>`  loss = loss_fct(...)` | 无 | vLLM纯推理，不计算损失 |
| **返回值** | `return SequenceClassifierOutputWithPast(...)` | `return logits` | vLLM返回简单张量 |

### 3.3 线性层映射规则

```python
# 映射规则表
┌────────────────────────────────────────────────────────────────────┐
│ 位置               HuggingFace              vLLM                    │
├────────────────────────────────────────────────────────────────────┤
│ 第一层/中间层      nn.Linear(in, out)       ColumnParallelLinear    │
│                                             - 跨GPU按列切分          │
│                                             - 输出(tensor, bias)     │
├────────────────────────────────────────────────────────────────────┤
│ 最后一层           nn.Linear(in, out)       RowParallelLinear       │
│                                             - 跨GPU按行切分          │
│                                             - 需要AllReduce         │
│                                             - 输出(tensor, bias)     │
├────────────────────────────────────────────────────────────────────┤
│ 激活函数           nn.ReLU()                自定义ReLU()            │
│                                             - 解包元组输入          │
│                                             - 只处理tensor部分       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. 常见转换模式

### 模式 1: 序列分类模型

```python
# ========== HuggingFace 模式 ==========
class ModelForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BaseModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask, ...):
        outputs = self.model(input_ids, attention_mask, ...)
        hidden = outputs[0][:, -1, :]  # 取最后一个token
        logits = self.classifier(hidden)
        return SequenceClassifierOutput(logits=logits, ...)

# ========== vLLM 模式 ==========
class ModelForSequenceClassification(nn.Module):
    def __init__(self, config, cache_config, quant_config, lora_config):
        super().__init__()
        self.model = BaseModel(config, cache_config, quant_config)
        self.classifier = RowParallelLinear(  # ← 使用行并行
            config.hidden_size,
            config.num_labels,
            quant_config=quant_config,
        )
        self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
    
    def forward(self, input_ids, positions, kv_caches, attn_metadata, ...):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        logits, _ = self.classifier(hidden_states)  # ← 解包元组
        return logits
    
    def pooler(self, hidden_states, pooling_metadata):
        return self._pooler(hidden_states, pooling_metadata)
```

### 模式 2: 多层MLP头

```python
# ========== HuggingFace 模式 ==========
self.head = nn.Sequential(
    nn.Linear(hidden, intermediate),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(intermediate, output),
)

# ========== vLLM 模式 ==========
# 方法1: 如果不需要dropout (推理时)
self.head = nn.Sequential(
    ColumnParallelLinear(hidden, intermediate, quant_config=quant_config),
    nn.GELU(),  # 标准激活可直接用
    RowParallelLinear(intermediate, output, quant_config=quant_config),
)

# 方法2: 如果需要处理dropout
class CustomGELU(nn.Module):
    def forward(self, input):
        input, _ = input
        return nn.functional.gelu(input)

self.head = nn.Sequential(
    ColumnParallelLinear(hidden, intermediate, quant_config=quant_config),
    CustomGELU(),
    RowParallelLinear(intermediate, output, quant_config=quant_config),
)
```

### 模式 3: 奖励模型 (完整示例)

```python
# ========== HuggingFace 版本 ==========
class RewardModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Transformer(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        self.post_init()
    
    def forward(self, input_ids, attention_mask, labels=None, ...):
        outputs = self.model(input_ids, attention_mask=attention_mask, ...)
        hidden = outputs[0]
        logits = self.score(hidden)
        
        # 找到最后一个非padding token
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.config.pad_token_id)
                .int().argmax(-1) - 1
            )
        
        pooled_logits = logits[torch.arange(batch_size), sequence_lengths]
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(pooled_logits, labels)
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
        )

# ========== vLLM 版本 ==========
class RewardModel(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    supported_lora_modules = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    
    def __init__(self, config, cache_config, quant_config, lora_config):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        
        self.model = Transformer(config, cache_config, quant_config)
        
        # 自定义ReLU处理元组
        class ReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.ReLU()
            def forward(self, input):
                input, _ = input
                return self.activation(input)
        
        self.score = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size,
                quant_config=quant_config,
            ),
            ReLU(),
            RowParallelLinear(
                config.hidden_size,
                1,
                quant_config=quant_config,
            ),
        )
        
        self._pooler = Pooler(pooling_type=PoolingType.ALL, normalize=False)
    
    def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors
        )
        logits, _ = self.score(hidden_states)
        return logits
    
    def pooler(self, hidden_states, pooling_metadata):
        return self._pooler(hidden_states, pooling_metadata)
    
    def load_weights(self, weights):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        for name, loaded_weight in weights:
            if name == "lm_head.weight" or "rotary_emb.inv_freq" in name:
                continue
            
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
```

---

## 5. 完整转换模板

```python
# ═══════════════════════════════════════════════════════════════════
# vLLM 模型实现模板
# ═══════════════════════════════════════════════════════════════════

from typing import Iterable, List, Optional, Tuple
import torch
from torch import nn

# vLLM imports
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name
)
from vllm.model_executor.models.xxx import BaseModel  # 替换为实际骨干网络
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from .utils import is_pp_missing_parameter


# ──────────────────────────────────────────────────────────────────
# 步骤1: 自定义激活函数 (如果使用并行层的话)
# ──────────────────────────────────────────────────────────────────
class CustomActivation(nn.Module):
    """
    包装标准激活函数以处理并行层的元组输出
    根据需要替换为 ReLU, GELU, SiLU 等
    """
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()  # 或 nn.GELU(), nn.SiLU() 等
    
    def forward(self, input):
        # 解包并行层的 (tensor, bias) 元组
        if isinstance(input, tuple):
            input, _ = input
        return self.activation(input)


# ──────────────────────────────────────────────────────────────────
# 步骤2: 主模型类
# ──────────────────────────────────────────────────────────────────
class YourModelForTask(nn.Module):
    """
    [模型名称] - vLLM优化实现
    
    用途: [描述模型的任务，如分类/回归/奖励模型等]
    
    原始实现: [HuggingFace模型链接]
    """
    
    # ═══════════════════════════════════════════════════════════════
    # 类属性配置
    # ═══════════════════════════════════════════════════════════════
    
    # 1. 参数堆叠映射 (用于权重加载)
    # 根据骨干网络类型调整，常见的有:
    packed_modules_mapping = {
        # Attention层的QKV合并
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        # FFN层的gate和up合并 (用于SwiGLU)
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    # 2. LoRA支持的模块列表
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # 如果任务头也支持LoRA，添加相应模块
        # "classifier",
    ]
    
    # 3. Embedding相关模块 (通常为空，除非有特殊embedding)
    embedding_modules = {}
    embedding_padding_modules = []
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化方法
    # ═══════════════════════════════════════════════════════════════
    def __init__(
        self,
        config,                                  # 模型配置对象
        cache_config: Optional[CacheConfig] = None,      # KV缓存配置
        quant_config: Optional[QuantizationConfig] = None,  # 量化配置
        lora_config: Optional[LoRAConfig] = None,        # LoRA配置
    ) -> None:
        super().__init__()
        
        # ───────────────────────────────────────────────────────────
        # 1. 保存配置
        # ───────────────────────────────────────────────────────────
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        
        # ───────────────────────────────────────────────────────────
        # 2. 初始化骨干网络
        # ───────────────────────────────────────────────────────────
        # 替换为实际的骨干网络类，如:
        # - Qwen2Model, LlamaModel, MistralModel 等
        self.model = BaseModel(config, cache_config, quant_config)
        
        # ───────────────────────────────────────────────────────────
        # 3. 初始化任务特定的头部
        # ───────────────────────────────────────────────────────────
        
        # 示例A: 单层分类头
        # self.classifier = RowParallelLinear(
        #     config.hidden_size,
        #     config.num_labels,
        #     quant_config=quant_config,
        # )
        
        # 示例B: 多层MLP头
        # self.head = nn.Sequential(
        #     ColumnParallelLinear(
        #         config.hidden_size,
        #         config.intermediate_size,
        #         quant_config=quant_config,
        #     ),
        #     CustomActivation(),
        #     RowParallelLinear(
        #         config.intermediate_size,
        #         config.num_labels,
        #         quant_config=quant_config,
        #     ),
        # )
        
        # 示例C: 奖励模型头 (两层MLP)
        self.score = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size,
                quant_config=quant_config,
            ),
            CustomActivation(),
            RowParallelLinear(
                config.hidden_size,
                1,  # 输出维度，根据任务调整
                quant_config=quant_config,
            ),
        )
        
        # ───────────────────────────────────────────────────────────
        # 4. 初始化Pooler
        # ───────────────────────────────────────────────────────────
        # 根据任务选择池化类型:
        # - PoolingType.LAST: 取最后一个token (序列分类)
        # - PoolingType.ALL: 保留所有token (token分类/生成)
        # - PoolingType.CLS: 取CLS token
        self._pooler = Pooler(
            pooling_type=PoolingType.LAST,  # 根据任务调整
            normalize=False,  # 是否L2归一化
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 前向传播方法
    # ═══════════════════════════════════════════════════════════════
    def forward(
        self,
        input_ids: torch.Tensor,                            # [batch, seq_len]
        positions: torch.Tensor,                            # [batch, seq_len]
        kv_caches: List[torch.Tensor],                     # List of KV caches
        attn_metadata: AttentionMetadata,                  # 注意力元数据
        intermediate_tensors: Optional[IntermediateTensors] = None,  # 流水线并行
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入token ID
            positions: token位置索引
            kv_caches: 每层的KV缓存
            attn_metadata: 注意力计算元数据
            intermediate_tensors: 流水线并行的中间张量
        
        Returns:
            torch.Tensor: 模型输出 (具体形状根据任务而定)
        """
        # ───────────────────────────────────────────────────────────
        # 1. 通过骨干网络提取特征
        # ───────────────────────────────────────────────────────────
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors
        )
        # hidden_states: [batch, seq_len, hidden_size]
        
        # ───────────────────────────────────────────────────────────
        # 2. 通过任务头计算输出
        # ───────────────────────────────────────────────────────────
        # 注意: 并行层返回 (tensor, bias) 元组，需要解包
        logits, _ = self.score(hidden_states)
        # logits: [batch, seq_len, output_dim]
        
        # ───────────────────────────────────────────────────────────
        # 3. 返回logits (池化在pooler方法中单独处理)
        # ───────────────────────────────────────────────────────────
        return logits
    
    # ═══════════════════════════════════════════════════════════════
    # Pooler方法 (提取最终输出)
    # ═══════════════════════════════════════════════════════════════
    def pooler(
        self,
        hidden_states: torch.Tensor,                       # 模型输出
        pooling_metadata: PoolingMetadata,                 # 池化元数据
    ) -> Optional[PoolerOutput]:
        """
        池化操作：从序列输出中提取最终结果
        
        Args:
            hidden_states: forward方法的输出
            pooling_metadata: 包含序列长度、边界等信息
        
        Returns:
            PoolerOutput: 池化后的输出
        """
        return self._pooler(hidden_states, pooling_metadata)
    
    # ═══════════════════════════════════════════════════════════════
    # 权重加载方法
    # ═══════════════════════════════════════════════════════════════
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        从HuggingFace checkpoint加载权重
        
        处理:
        1. 参数名映射 (HF → vLLM)
        2. 堆叠参数的正确加载
        3. 并行切分
        4. 量化参数处理
        
        Args:
            weights: 迭代器，产生 (参数名, 权重张量) 对
        """
        # ───────────────────────────────────────────────────────────
        # 1. 定义堆叠参数映射
        # ───────────────────────────────────────────────────────────
        # 格式: (vLLM参数名, HF参数名, shard标识)
        stacked_params_mapping = [
            # QKV堆叠
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # Gate-Up堆叠
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        # ───────────────────────────────────────────────────────────
        # 2. 获取模型参数字典
        # ───────────────────────────────────────────────────────────
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        # ───────────────────────────────────────────────────────────
        # 3. 遍历并加载权重
        # ───────────────────────────────────────────────────────────
        for name, loaded_weight in weights:
            # ──────────────────────────────────────────────────────
            # 3.1 跳过不需要的权重
            # ──────────────────────────────────────────────────────
            
            # 跳过语言模型头 (如果任务不需要)
            if name == "lm_head.weight":
                continue
            
            # 跳过旋转位置编码的计算参数
            if "rotary_emb.inv_freq" in name:
                continue
            
            # ──────────────────────────────────────────────────────
            # 3.2 处理堆叠参数
            # ──────────────────────────────────────────────────────
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                
                # 替换参数名
                name = name.replace(weight_name, param_name)
                
                # 跳过不存在的bias (GPTQ量化模型)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                # 检查流水线并行 (是否属于当前stage)
                if is_pp_missing_parameter(name, self):
                    continue
                
                # 获取参数并加载
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # ──────────────────────────────────────────────────
                # 3.3 处理常规参数
                # ──────────────────────────────────────────────────
                
                # 跳过GPTQ bias
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                # 重映射FP8量化的kv_scale参数
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                
                # 检查流水线并行
                if is_pp_missing_parameter(name, self):
                    continue
                
                # 加载权重
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader
                )
                weight_loader(param, loaded_weight)


# ═══════════════════════════════════════════════════════════════════
# 转换检查清单
# ═══════════════════════════════════════════════════════════════════
"""
转换完成后，请检查以下项目:

✅ 类定义
   □ 继承自 nn.Module (不是 PreTrainedModel)
   □ 添加了 packed_modules_mapping
   □ 添加了 supported_lora_modules
   □ 添加了 embedding_modules (通常为空)

✅ __init__ 方法
   □ 参数包含: config, cache_config, quant_config, lora_config
   □ 保存了所有配置到 self
   □ 骨干网络传入了 cache_config 和 quant_config
   □ 使用 ColumnParallelLinear / RowParallelLinear
   □ 自定义激活函数处理元组输入
   □ 初始化了 self._pooler

✅ forward 方法
   □ 参数: input_ids, positions, kv_caches, attn_metadata, intermediate_tensors
   □ 移除了训练相关参数 (labels, return_dict 等)
   □ 骨干网络调用传入了所有vLLM参数
   □ 解包并行层的元组输出: logits, _ = self.score(...)
   □ 返回简单张量 (不是字典或对象)

✅ pooler 方法
   □ 实现了 pooler 方法
   □ 接受 hidden_states 和 pooling_metadata
   □ 调用 self._pooler 并返回结果

✅ load_weights 方法
   □ 定义了 stacked_params_mapping
   □ 获取了 params_dict
   □ 跳过 lm_head.weight (如果不需要)
   □ 跳过 rotary_emb.inv_freq
   □ 处理堆叠参数映射
   □ 处理常规参数
   □ 调用 weight_loader 加载权重

✅ 测试
   □ 能够成功加载HuggingFace权重
   □ 推理输出形状正确
   □ 支持tensor并行 (多GPU)
   □ 支持量化 (如果配置了)
   □ 性能符合预期
"""

# ═══════════════════════════════════════════════════════════════════
# 6. 转换实战：常见问题与解决方案
# ═══════════════════════════════════════════════════════════════════

## 问题1: 如何处理多输出头？

### HuggingFace实现
```python
class ModelWithMultiHeads(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BaseModel(config)
        self.head1 = nn.Linear(config.hidden_size, config.num_labels_1)
        self.head2 = nn.Linear(config.hidden_size, config.num_labels_2)
    
    def forward(self, input_ids, ...):
        hidden = self.model(input_ids, ...)
        output1 = self.head1(hidden)
        output2 = self.head2(hidden)
        return (output1, output2)
```

### vLLM实现
```python
class ModelWithMultiHeads(nn.Module):
    def __init__(self, config, cache_config, quant_config, lora_config):
        super().__init__()
        self.config = config
        self.model = BaseModel(config, cache_config, quant_config)
        
        # 两个头都使用并行层
        self.head1 = RowParallelLinear(
            config.hidden_size,
            config.num_labels_1,
            quant_config=quant_config,
        )
        self.head2 = RowParallelLinear(
            config.hidden_size,
            config.num_labels_2,
            quant_config=quant_config,
        )
        
        # 可以使用不同的pooler
        self._pooler1 = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        self._pooler2 = Pooler(pooling_type=PoolingType.MEAN, normalize=True)
    
    def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)
        
        # 解包两个头的输出
        output1, _ = self.head1(hidden_states)
        output2, _ = self.head2(hidden_states)
        
        # 可以返回拼接的张量或字典
        return torch.cat([output1, output2], dim=-1)
    
    def pooler(self, hidden_states, pooling_metadata):
        # 如果需要不同的池化策略，可以分别处理
        # 这里简化为使用第一个pooler
        return self._pooler1(hidden_states, pooling_metadata)
```

## 问题2: 如何处理Dropout层？

**重要**: vLLM是纯推理框架，训练时的Dropout在推理时应该被忽略

### HuggingFace实现
```python
self.classifier = nn.Sequential(
    nn.Linear(hidden, intermediate),
    nn.Dropout(0.1),  # 训练时使用
    nn.Linear(intermediate, output),
)
```

### vLLM实现 - 方案A: 直接移除
```python
self.classifier = nn.Sequential(
    ColumnParallelLinear(hidden, intermediate, quant_config=quant_config),
    CustomActivation(),
    RowParallelLinear(intermediate, output, quant_config=quant_config),
)
# Dropout在推理时等价于恒等映射，直接移除
```

### vLLM实现 - 方案B: 保留但设置eval模式
```python
self.classifier = nn.Sequential(
    ColumnParallelLinear(hidden, intermediate, quant_config=quant_config),
    nn.Dropout(0.1),  # 保留，但模型会自动在eval模式下禁用
    RowParallelLinear(intermediate, output, quant_config=quant_config),
)
# vLLM会自动将模型设置为eval模式，Dropout会被禁用
```

## 问题3: 如何处理LayerNorm/RMSNorm？

标准的归一化层不需要特殊处理，直接使用即可：

```python
# HuggingFace和vLLM通用
self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
self.rms_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

# 在forward中正常使用
normalized = self.layer_norm(hidden_states)
```

## 问题4: 如何处理条件分支（if-else）？

### HuggingFace实现
```python
def forward(self, input_ids, task_type=None, ...):
    hidden = self.model(input_ids, ...)
    
    if task_type == "classification":
        output = self.classifier(hidden)
    elif task_type == "regression":
        output = self.regressor(hidden)
    else:
        output = self.default_head(hidden)
    
    return output
```

### vLLM实现
vLLM推理时通常只需要一个任务路径，所以：

**方案A: 固定任务类型**
```python
def __init__(self, config, cache_config, quant_config, lora_config):
    super().__init__()
    # 只初始化需要的头
    task_type = getattr(config, "task_type", "classification")
    
    if task_type == "classification":
        self.head = RowParallelLinear(...)
    elif task_type == "regression":
        self.head = RowParallelLinear(...)
    
def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
    hidden_states = self.model(...)
    output, _ = self.head(hidden_states)
    return output
```

**方案B: 保留所有分支（如果需要）**
```python
def __init__(self, config, cache_config, quant_config, lora_config):
    super().__init__()
    self.classifier = RowParallelLinear(...)
    self.regressor = RowParallelLinear(...)

def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
    hidden_states = self.model(...)
    
    # 通过配置或外部信号决定使用哪个头
    if self.config.task_type == "classification":
        output, _ = self.classifier(hidden_states)
    else:
        output, _ = self.regressor(hidden_states)
    
    return output
```

## 问题5: 如何处理自定义的初始化？

### HuggingFace实现
```python
def __init__(self, config):
    super().__init__(config)
    self.model = BaseModel(config)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    # 自定义初始化
    self.post_init()

def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
```

### vLLM实现
vLLM从checkpoint加载权重，不需要随机初始化：

```python
def __init__(self, config, cache_config, quant_config, lora_config):
    super().__init__()
    self.config = config
    self.model = BaseModel(config, cache_config, quant_config)
    self.classifier = RowParallelLinear(
        config.hidden_size,
        config.num_labels,
        quant_config=quant_config,
    )
    
    # 不需要 post_init() 或 _init_weights()
    # 权重会通过 load_weights() 从checkpoint加载
```

## 问题6: 如何处理嵌套的Sequential？

### HuggingFace实现
```python
self.layers = nn.ModuleList([
    nn.Sequential(
        nn.Linear(hidden, intermediate),
        nn.ReLU(),
        nn.Linear(intermediate, hidden),
    )
    for _ in range(num_layers)
])
```

### vLLM实现
```python
# 自定义块类处理元组
class TransformBlock(nn.Module):
    def __init__(self, hidden, intermediate, quant_config):
        super().__init__()
        self.up = ColumnParallelLinear(hidden, intermediate, quant_config=quant_config)
        self.act = CustomActivation()
        self.down = RowParallelLinear(intermediate, hidden, quant_config=quant_config)
    
    def forward(self, x):
        x, _ = self.up(x)
        x = self.act((x, None))  # 包装为元组给激活函数
        x, _ = self.down(x)
        return x

# 使用自定义块
self.layers = nn.ModuleList([
    TransformBlock(hidden, intermediate, quant_config)
    for _ in range(num_layers)
])
```

## 问题7: 如何处理自定义的前向逻辑（如残差连接）？

### HuggingFace实现
```python
def forward(self, input_ids, ...):
    hidden = self.model(input_ids, ...)
    
    # 复杂的前向逻辑
    residual = hidden
    hidden = self.norm(hidden)
    hidden = self.ffn(hidden)
    hidden = residual + hidden
    
    output = self.head(hidden)
    return output
```

### vLLM实现
完全保留相同的逻辑，只需注意解包：

```python
def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
    hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)
    
    # 相同的前向逻辑
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    hidden_states, _ = self.ffn(hidden_states)  # ← 注意解包
    hidden_states = residual + hidden_states
    
    output, _ = self.head(hidden_states)  # ← 注意解包
    return output
```

# ═══════════════════════════════════════════════════════════════════
# 7. 调试技巧
# ═══════════════════════════════════════════════════════════════════

## 技巧1: 对比输出形状

```python
# 在HuggingFace模型中
print(f"HF hidden_states shape: {hidden_states.shape}")
print(f"HF logits shape: {logits.shape}")

# 在vLLM模型中
print(f"vLLM hidden_states shape: {hidden_states.shape}")
print(f"vLLM logits shape: {logits.shape}")

# 应该完全一致！
```

## 技巧2: 验证权重加载

```python
# 加载HuggingFace权重后，检查参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# 确保:
# 1. 所有参数都被加载了
# 2. 堆叠参数的形状正确 (如qkv_proj应该是3倍的hidden_size)
# 3. 没有未初始化的参数
```

## 技巧3: 对比数值输出

```python
import torch

# 准备相同的输入
input_ids = torch.randint(0, 1000, (2, 10))  # [batch=2, seq_len=10]

# HuggingFace推理
hf_model.eval()
with torch.no_grad():
    hf_output = hf_model(input_ids)
    hf_logits = hf_output.logits

# vLLM推理 (需要准备vLLM格式的输入)
vllm_model.eval()
with torch.no_grad():
    positions = torch.arange(10).unsqueeze(0).expand(2, -1)
    # ... 准备kv_caches和attn_metadata
    vllm_output = vllm_model(input_ids, positions, kv_caches, attn_metadata)

# 对比结果
print(f"Output difference: {torch.abs(hf_logits - vllm_output).max().item()}")
# 应该非常小 (< 1e-4)
```

## 技巧4: 逐层验证

```python
# 在forward中添加调试输出
def forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors=None):
    print(f"[DEBUG] input_ids shape: {input_ids.shape}")
    
    hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors)
    print(f"[DEBUG] hidden_states shape: {hidden_states.shape}")
    print(f"[DEBUG] hidden_states range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    
    logits, bias = self.score(hidden_states)
    print(f"[DEBUG] logits shape: {logits.shape}")
    print(f"[DEBUG] bias shape: {bias.shape if bias is not None else None}")
    
    return logits
```

## 技巧5: 检查并行层的输出

```python
# 测试并行层是否正确处理元组
class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ColumnParallelLinear(768, 768)
        self.act = CustomActivation()
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        out = self.linear(x)
        print(f"ColumnParallel output type: {type(out)}")
        print(f"ColumnParallel output[0] shape: {out[0].shape if isinstance(out, tuple) else 'not tuple'}")
        
        activated = self.act(out)
        print(f"After activation shape: {activated.shape}")
        
        return activated

# 运行测试
test = TestModule()
x = torch.randn(2, 10, 768)
output = test(x)
```

# ═══════════════════════════════════════════════════════════════════
# 8. 完整转换实例：从零开始
# ═══════════════════════════════════════════════════════════════════

让我们通过一个完整的例子，演示如何将一个简单的HuggingFace分类模型转换为vLLM实现。

## 原始HuggingFace实现

```python
# ========== original_model.py (HuggingFace) ==========
from transformers import PreTrainedModel, LlamaModel
import torch
import torch.nn as nn

class SentimentClassifier(PreTrainedModel):
    """情感分类模型 - HuggingFace实现"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 骨干网络
        self.model = LlamaModel(config)
        
        # 分类头: 2层MLP
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, self.num_labels)
        )
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 1. 通过骨干网络提取特征
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 2. 获取序列表示
        hidden_states = outputs[0]  # [batch, seq_len, hidden_size]
        
        # 3. 提取最后一个token (用于分类)
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1)
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(hidden_states.device)
            else:
                sequence_lengths = -1
        
        # 4. 池化
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        # 5. Dropout
        pooled_hidden_states = self.dropout(pooled_hidden_states)
        
        # 6. 分类
        logits = self.classifier(pooled_hidden_states)
        
        # 7. 计算损失
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 8. 返回结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import SequenceClassifierOutputWithPast
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

## 转换后的vLLM实现

```python
# ========== vllm_model.py (vLLM) ==========
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

# vLLM imports
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name
)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput

from .utils import is_pp_missing_parameter


# ──────────────────────────────────────────────────────────────────
# 步骤1: 自定义Tanh激活 (处理并行层的元组输出)
# ──────────────────────────────────────────────────────────────────
class TanhActivation(nn.Module):
    """自定义Tanh激活，处理并行层的元组输出"""
    
    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()
    
    def forward(self, input):
        # 如果输入是元组 (来自ColumnParallelLinear)
        if isinstance(input, tuple):
            input, _ = input
        return self.activation(input)


# ──────────────────────────────────────────────────────────────────
# 步骤2: vLLM优化的情感分类模型
# ──────────────────────────────────────────────────────────────────
class SentimentClassifier(nn.Module):
    """
    情感分类模型 - vLLM优化实现
    
    原始实现: HuggingFace版本的SentimentClassifier
    优化: 支持tensor并行、量化、高吞吐量批量推理
    """
    
    # ═══════════════════════════════════════════════════════════════
    # 类属性配置
    # ═══════════════════════════════════════════════════════════════
    
    # Llama模型的参数堆叠映射
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    # LoRA支持的模块
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # 分类头也可以支持LoRA
        "classifier.0",  # 第一个线性层
        "classifier.2",  # 第二个线性层
    ]
    
    embedding_modules = {}
    embedding_padding_modules = []
    
    # ═══════════════════════════════════════════════════════════════
    # 初始化
    # ═══════════════════════════════════════════════════════════════
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        
        # 保存配置
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.num_labels = config.num_labels
        
        # 初始化骨干网络 (Llama)
        self.model = LlamaModel(config, cache_config, quant_config)
        
        # 初始化分类头
        # 注意:
        # 1. 不使用Dropout (推理时不需要)
        # 2. 使用并行线性层
        # 3. 使用自定义激活函数
        self.classifier = nn.Sequential(
            # 第一层: 降维 (hidden_size → hidden_size // 2)
            ColumnParallelLinear(
                config.hidden_size,
                config.hidden_size // 2,
                bias=True,
                quant_config=quant_config,
            ),
            # Tanh激活
            TanhActivation(),
            # 第二层: 分类 (hidden_size // 2 → num_labels)
            RowParallelLinear(
                config.hidden_size // 2,
                self.num_labels,
                bias=True,
                quant_config=quant_config,
            ),
        )
        
        # 初始化Pooler (取最后一个token)
        self._pooler = Pooler(
            pooling_type=PoolingType.LAST,
            normalize=False,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # 前向传播
    # ═══════════════════════════════════════════════════════════════
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入token IDs
            positions: [batch_size, seq_len] 位置索引
            kv_caches: KV缓存列表
            attn_metadata: 注意力元数据
            intermediate_tensors: 流水线并行的中间张量
        
        Returns:
            logits: [batch_size, seq_len, num_labels] 分类logits
        """
        # 1. 通过骨干网络提取特征
        hidden_states = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors
        )
        # hidden_states: [batch_size, seq_len, hidden_size]
        
        # 2. 通过分类头计算logits
        # 注意: 最后一层是RowParallelLinear，返回元组
        logits, _ = self.classifier(hidden_states)
        # logits: [batch_size, seq_len, num_labels]
        
        return logits
    
    # ═══════════════════════════════════════════════════════════════
    # Pooler (提取最后一个token的logits)
    # ═══════════════════════════════════════════════════════════════
    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        """
        池化: 从序列输出中提取最后一个有效token的logits
        
        Args:
            hidden_states: [total_tokens, num_labels] 所有token的logits
            pooling_metadata: 池化元数据 (包含序列边界信息)
        
        Returns:
            PoolerOutput: 包含每个序列最后一个token的logits
        """
        return self._pooler(hidden_states, pooling_metadata)
    
    # ═══════════════════════════════════════════════════════════════
    # 权重加载
    # ═══════════════════════════════════════════════════════════════
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        从HuggingFace checkpoint加载权重
        
        Args:
            weights: 迭代器，产生 (参数名, 权重张量) 对
        """
        # 定义参数堆叠映射
        stacked_params_mapping = [
            # Llama的QKV堆叠
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # Llama的gate-up堆叠
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        
        # 获取参数字典
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        # 遍历权重
        for name, loaded_weight in weights:
            # 跳过不需要的权重
            if name == "lm_head.weight":
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            
            # 处理堆叠参数
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                
                name = name.replace(weight_name, param_name)
                
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                if is_pp_missing_parameter(name, self):
                    continue
                
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 处理常规参数
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                
                if is_pp_missing_parameter(name, self):
                    continue
                
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader
                )
                weight_loader(param, loaded_weight)
```

## 转换对照表

| 组件 | HuggingFace | vLLM | 变化说明 |
|------|-------------|------|---------|
| **基类** | `PreTrainedModel` | `nn.Module` | 移除HF特定功能 |
| **__init__参数** | `config` | `config, cache_config, quant_config, lora_config` | 新增推理配置 |
| **骨干网络** | `LlamaModel(config)` | `LlamaModel(config, cache_config, quant_config)` | 传入推理配置 |
| **Dropout** | `nn.Dropout(0.1)` | 移除 | 推理时不需要 |
| **第1层Linear** | `nn.Linear(hidden, hidden//2)` | `ColumnParallelLinear(...)` | 支持并行 |
| **Tanh** | `nn.Tanh()` | `TanhActivation()` | 处理元组输入 |
| **第2层Linear** | `nn.Linear(hidden//2, num_labels)` | `RowParallelLinear(...)` | 支持并行 |
| **forward参数** | 11个参数 (labels, return_dict等) | 5个参数 (vLLM专用) | 简化为推理参数 |
| **池化逻辑** | 在forward内部 | 独立pooler方法 | 逻辑解耦 |
| **损失计算** | 在forward中计算 | 移除 | 纯推理 |
| **返回值** | `SequenceClassifierOutputWithPast` | `torch.Tensor` | 简化输出 |
| **权重初始化** | `post_init()` | 无 | 从checkpoint加载 |
| **新增方法** | 无 | `load_weights()` | 自定义权重加载 |

# ═══════════════════════════════════════════════════════════════════
# 9. 注册模型到vLLM
# ═══════════════════════════════════════════════════════════════════

转换完成后，需要将模型注册到vLLM框架：

## 步骤1: 创建模型文件

```bash
# vLLM模型文件结构
vllm/model_executor/models/
├── __init__.py
├── llama.py
├── qwen2.py
└── sentiment_classifier.py  # ← 你的新模型
```

## 步骤2: 在__init__.py中注册

```python
# vllm/model_executor/models/__init__.py

# ... 其他导入 ...

# 注册你的模型
_MODEL_REGISTRY = {
    # ... 现有模型 ...
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    
    # 新增你的模型
    "SentimentClassifier": ("sentiment_classifier", "SentimentClassifier"),
}
```

## 步骤3: 配置模型架构

```python
# config.json 中指定架构
{
  "architectures": [
    "SentimentClassifier"  # ← 与注册名称匹配
  ],
  "model_type": "llama",  # 基于的骨干网络类型
  ...
}
```

## 步骤4: 使用模型

```python
from vllm import LLM

# 初始化模型
llm = LLM(
    model="path/to/your/model",
    task="classify",  # 或 "embed"
    tensor_parallel_size=2,  # 使用2个GPU
    quantization="fp8",  # 可选: 使用FP8量化
)

# 推理
prompts = [
    "This movie is amazing!",
    "I hate this product.",
]

outputs = llm.encode(prompts)

for output in outputs:
    print(f"Text: {output.prompt}")
    print(f"Logits: {output.outputs[0]}")  # [num_labels]的向量
```

# ═══════════════════════════════════════════════════════════════════
# 10. 性能优化建议
# ═══════════════════════════════════════════════════════════════════

## 优化1: 使用FP8量化

```python
# 启用FP8量化可以显著提升吞吐量
llm = LLM(
    model="your-model",
    quantization="fp8",  # 或 "int8", "gptq"
    tensor_parallel_size=4,
)

# 性能提升: 2-3x吞吐量, 50%内存占用
```

## 优化2: 调整批次大小

```python
# 增大max_num_seqs以提高吞吐量
llm = LLM(
    model="your-model",
    max_num_seqs=256,  # 默认是256，可以增加
    max_model_len=512,  # 根据你的序列长度调整
)
```

## 优化3: 启用连续批处理

vLLM自动启用continuous batching，无需配置。这允许：
- 动态添加新请求到批次
- 不同长度的序列高效处理
- 最大化GPU利用率

## 优化4: 使用PagedAttention

vLLM自动使用PagedAttention管理KV缓存：
- 减少内存碎片
- 支持更大的批次
- 接近零内存浪费

## 优化5: Tensor并行最佳实践

```python
# GPU数量选择建议:
# - 小模型 (< 13B): tensor_parallel_size=1
# - 中等模型 (13B-40B): tensor_parallel_size=2 或 4
# - 大模型 (> 40B): tensor_parallel_size=4 或 8

# 确保模型能均匀切分
assert config.hidden_size % tensor_parallel_size == 0
assert config.num_attention_heads % tensor_parallel_size == 0
```

# ═══════════════════════════════════════════════════════════════════
# 11. 常见错误与解决方案
# ═══════════════════════════════════════════════════════════════════

## 错误1: 形状不匹配

```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
```

**原因**: 并行层返回元组，但直接当作张量使用

**解决**:
```python
# ❌ 错误
output = self.layer(input)
result = output + residual  # output是元组！

# ✅ 正确
output, _ = self.layer(input)
result = output + residual
```

## 错误2: 权重加载失败

```
KeyError: 'qkv_proj.weight' not found in state dict
```

**原因**: 参数名映射不正确

**解决**:
检查`stacked_params_mapping`是否包含所有需要堆叠的参数：
```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),  # ← 确保这些映射正确
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]
```

## 错误3: PoolingType不匹配

```
ValueError: Cannot pool sequences with different lengths
```

**原因**: Pooling类型选择不当

**解决**:
```python
# 序列分类: 使用LAST
self._pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

# Token分类: 使用ALL
self._pooler = Pooler(pooling_type=PoolingType.ALL, normalize=False)

# 句子嵌入: 使用MEAN + 归一化
self._pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=True)
```

## 错误4: CUDA OOM

```
RuntimeError: CUDA out of memory
```

**解决方案**:
1. 减小batch size: `max_num_seqs=128`
2. 减小序列长度: `max_model_len=512`
3. 启用量化: `quantization="fp8"`
4. 增加GPU数量: `tensor_parallel_size=4`

## 错误5: 激活函数错误

```
TypeError: forward() takes 2 positional arguments but 3 were given
```

**原因**: 标准激活函数不能处理元组输入

**解决**:
```python
# ❌ 错误
self.act = nn.ReLU()

# ✅ 正确
class CustomReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
    
    def forward(self, input):
        if isinstance(input, tuple):
            input, _ = input
        return self.activation(input)

self.act = CustomReLU()
```

# ═══════════════════════════════════════════════════════════════════
# 12. 转换总结与检查清单
# ═══════════════════════════════════════════════════════════════════

## ✅ 转换前检查清单

- [ ] 理解原始HF模型的架构
- [ ] 识别所有线性层的位置
- [ ] 识别所有激活函数
- [ ] 识别池化逻辑
- [ ] 了解模型的输入输出格式
- [ ] 检查是否有条件分支
- [ ] 检查是否有自定义初始化

## ✅ 转换过程检查清单

- [ ] 将基类改为`nn.Module`
- [ ] 添加`packed_modules_mapping`
- [ ] 添加`supported_lora_modules`
- [ ] 修改`__init__`参数（增加cache_config等）
- [ ] 将`nn.Linear`替换为`ColumnParallelLinear`/`RowParallelLinear`
- [ ] 包装激活函数以处理元组
- [ ] 移除Dropout层（或保留但不影响推理）
- [ ] 修改`forward`签名（使用vLLM参数）
- [ ] 解包所有并行层的输出
- [ ] 将池化逻辑移到`pooler`方法
- [ ] 移除损失计算和标签处理
- [ ] 简化返回值（直接返回张量）
- [ ] 实现`load_weights`方法
- [ ] 添加参数堆叠映射逻辑

## ✅ 转换后