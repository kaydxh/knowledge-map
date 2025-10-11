## HuggingFace 到 vLLM 实现转换完整指南
### 1 核心对应关系
#### 1.1 整体架构映射

```
HuggingFace 实现                          vLLM 实现
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────┐    ┌──────────────────────────────┐
│ class Qwen2ForRewardModel       │    │ class Qwen2ForRewardModel    │
│   (PreTrainedModel)             │ →  │   (nn.Module)                │
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
│   → 返回 SequenceClassifier     │     │   → 返回 torch.Tensor        │
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
#### 1.2 逐层对应关系
| 组件 | HuggingFace | vLLM | 关键差异|
|--------|--------|------|-------|
|   基类  | PreTrainedModel | nn.Module | vLLM不需要HF的模型管理功能 |
|  初始化参数 | config  | config, cache_config, quant_config, lora_config | vLLM需要推理相关配置 |
|   线性层   |    nn.Linear    |  ColumnParallelLinear / RowParallelLinear    |  vLLM支持tensor并行     |
|   激活函数  | nn.ReLU()   | ReLU() (自定义)  | vLLM需要处理元组输出    |
|前向参数 | 训练友好(labels, return_dict)|推理优化(kv_caches, attn_metadata) | 不同使用场景|
|输出格式 |结构化对象 | 原始张量 | vLLM追求性能|
|池化 | 内嵌在forward | 独立的pooler方法 | vLLM解耦逻辑 |
|权重加载 | | | |
