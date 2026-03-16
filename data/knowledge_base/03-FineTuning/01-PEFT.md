# 参数高效微调 (PEFT)

## 1. LoRA (Low-Rank Adaptation)

### Q: LoRA 的原理 ⭐⭐⭐⭐

**核心思想**：冻结预训练权重，只训练低秩分解矩阵。

$$W' = W + \Delta W = W + BA$$

其中：
- $W \in \mathbb{R}^{d \times d}$：原始权重（冻结）
- $B \in \mathbb{R}^{d \times r}$：低秩矩阵
- $A \in \mathbb{R}^{r \times d}$：低秩矩阵
- $r \ll d$：秩（通常 4~64）

**参数量对比**：
- 全量微调：$d \times d$ 参数
- LoRA：$d \times r + r \times d = 2dr$ 参数
- 当 $r=16, d=4096$ 时，参数量比约为 0.78%

### Q: LoRA 的秩 r 如何选择？

| 秩 r | 参数量 | 适用场景 |
|------|--------|----------|
| 4-8 | 极少 | 简单任务适配 |
| 16-32 | 少 | 通用推荐 |
| 64-128 | 中等 | 复杂任务/大模型 |
| 256+ | 较多 | 接近全量微调 |

### Q: LoRA 应用在哪些层？

通常应用在 Attention 的 Q, K, V, O 投影矩阵上。实验表明同时对 Q 和 V 做 LoRA 效果最佳。

---

## 2. QLoRA

### Q: QLoRA 的原理

在 LoRA 基础上，将基础模型量化到 4-bit（NF4），然后在量化模型上做 LoRA 微调。

**关键技术**：
1. **NF4 量化**：针对正态分布的最优 4-bit 量化
2. **Double Quantization**：对量化常数再次量化
3. **Paged Optimizers**：利用统一内存管理显存

**效果**：65B 模型可在单张 48GB GPU 上微调。

---

## 3. 其他 PEFT 方法

### Adapter

在 Transformer 层中插入小型 Adapter 模块：
```
Attention → Adapter(↓ → NonLinear → ↑) → FFN → Adapter(↓ → NonLinear → ↑)
```

### Prefix Tuning

在每层的 KV 前面添加可训练的 prefix：
$$K' = [P_K; K], \quad V' = [P_V; V]$$

### Prompt Tuning

在输入 embedding 前添加可训练的软提示：
$$E' = [P; E]$$

### 方法对比

| 方法 | 参数量 | 推理延迟 | 效果 | 推荐 |
|------|--------|----------|------|------|
| LoRA | 0.1-1% | 无增加 | ⭐⭐⭐⭐ | ✅ |
| QLoRA | 0.1-1% | 量化损失 | ⭐⭐⭐ | ✅ |
| Adapter | 1-5% | 有增加 | ⭐⭐⭐ | |
| Prefix | 0.1% | 有增加 | ⭐⭐ | |
| Prompt | <0.1% | 无增加 | ⭐⭐ | |
