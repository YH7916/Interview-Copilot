# Transformer 架构详解

## 1. Transformer 整体架构

### Q: 请描述 Transformer 的整体架构

Transformer 由 **Encoder** 和 **Decoder** 两部分组成（原始论文），每个部分由 N 个相同的层堆叠而成。

**Encoder 层结构：**
```
Input → Embedding + Positional Encoding
     → Multi-Head Self-Attention → Add & Norm
     → Feed-Forward Network → Add & Norm
     → Output
```

**Decoder 层结构：**
```
Input → Embedding + Positional Encoding
     → Masked Multi-Head Self-Attention → Add & Norm
     → Cross-Attention (with Encoder output) → Add & Norm
     → Feed-Forward Network → Add & Norm
     → Output
```

### Q: 为什么 Transformer 比 RNN 好？

| 特性 | RNN/LSTM | Transformer |
|------|----------|-------------|
| 并行化 | ❌ 序列依赖 | ✅ 完全并行 |
| 长距离依赖 | 梯度消失/爆炸 | 直接建模 |
| 计算复杂度 | O(n) 步 | O(1) 步（但 O(n²) 注意力） |
| 训练速度 | 慢 | 快 |

---

## 2. Self-Attention 机制

### Q: 请详细推导 Self-Attention 的计算过程 ⭐⭐⭐⭐⭐

**核心公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**计算步骤：**

1. **线性变换**：输入 $X \in \mathbb{R}^{n \times d}$，通过三个权重矩阵生成 Q、K、V

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$

2. **计算注意力分数**：

$$S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

3. **Softmax 归一化**：对每一行做 softmax

$$A = \text{softmax}(S) \in \mathbb{R}^{n \times n}$$

4. **加权求和**：

$$\text{Output} = AV \in \mathbb{R}^{n \times d_v}$$

### Q: 为什么要除以 $\sqrt{d_k}$？

当 $d_k$ 较大时，$QK^T$ 的值会很大（方差约为 $d_k$），导致 softmax 输出趋近于 one-hot 分布，梯度接近为零。

除以 $\sqrt{d_k}$ 可以使方差稳定在 1 左右，保证 softmax 有合理的梯度。

**数学证明**：假设 $q_i, k_j \sim \mathcal{N}(0, 1)$，则：

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

除以 $\sqrt{d_k}$ 后方差变为 1。

### Q: Self-Attention 的计算复杂度是多少？

| 操作 | 复杂度 |
|------|--------|
| $QK^T$ 计算 | $O(n^2 d)$ |
| Softmax | $O(n^2)$ |
| $\text{Softmax} \times V$ | $O(n^2 d)$ |
| **总计** | **$O(n^2 d)$** |

其中 $n$ 为序列长度，$d$ 为维度。

---

## 3. Multi-Head Attention (MHA)

### Q: Multi-Head Attention 的原理和作用 ⭐⭐⭐⭐⭐

**公式：**

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**参数关系**：
- 总维度 $d_{model}$，头数 $h$
- 每个头的维度 $d_k = d_v = d_{model} / h$
- 参数量与单头 Attention 相同

**作用**：
1. 允许模型在不同位置同时关注不同子空间的信息
2. 不同头可以学习不同的注意力模式（语法、语义、位置等）
3. 计算量与单头相同，但表达能力更强

### Q: MHA、MQA、GQA、MLA 的区别 ⭐⭐⭐⭐

| 方法 | 描述 | Q头数 | K/V头数 | KV Cache | 代表模型 |
|------|------|-------|---------|----------|----------|
| **MHA** | 标准多头 | $h$ | $h$ | 大 | GPT-2, BERT |
| **MQA** | 多查询注意力 | $h$ | 1 | 最小 | PaLM, Falcon |
| **GQA** | 分组查询注意力 | $h$ | $g$ ($1<g<h$) | 中等 | LLaMA-2 70B |
| **MLA** | 多头潜在注意力 | $h$ | 压缩潜在空间 | 最小 | DeepSeek-V2/V3 |

**GQA 原理**：将 $h$ 个 Q 头分成 $g$ 组，每组共享一个 KV 头。

**MLA 原理**（DeepSeek-V2）：
- 将 KV 压缩到低维潜在空间 $c_t = W_{DKV} [k_t; v_t]$
- 只缓存低维的 $c_t$，需要时再投影回来
- KV Cache 大幅减少（93.3%↓）

---

## 4. 位置编码

### Q: RoPE (旋转位置编码) 的原理 ⭐⭐⭐

**核心思想**：通过旋转矩阵将位置信息编码到 Q、K 中，使得内积只依赖于**相对位置**。

**二维情况**：

$$f(q, m) = R_m q = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$

**关键性质**：

$$\langle f(q, m), f(k, n) \rangle = q^T R_{n-m} k = g(q, k, n-m)$$

内积只依赖于相对位置 $n - m$。

**优势**：
- 相对位置编码，理论上可外推到更长序列
- 与线性注意力兼容
- 实现高效（逐元素乘法 + 旋转）

### Q: 各种位置编码对比

| 方法 | 类型 | 外推性 | 参数量 | 使用模型 |
|------|------|--------|--------|----------|
| Sinusoidal | 绝对 | 较差 | 0 | Transformer 原始 |
| Learned | 绝对 | 差 | $n \times d$ | GPT-2, BERT |
| ALiBi | 相对 | 好 | 0 | BLOOM |
| **RoPE** | **相对** | **较好** | **0** | **LLaMA, ChatGLM** |

---

## 5. LayerNorm & 归一化

### Q: LayerNorm vs BatchNorm 的区别

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | Batch 维度 | Feature 维度 |
| 依赖 batch size | ✅ 是 | ❌ 否 |
| 适用场景 | CV | NLP/Transformer |
| 推理时 | 需要 running stats | 直接计算 |

**LayerNorm 公式**：

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中 $\mu, \sigma$ 在最后一个维度上计算。

### Q: Pre-Norm vs Post-Norm

- **Post-Norm**（原始 Transformer）：$x + \text{LN}(\text{SubLayer}(x))$
  - 理论上表达能力更强
  - 训练不稳定，需要 warmup

- **Pre-Norm**（GPT-2/LLaMA）：$x + \text{SubLayer}(\text{LN}(x))$
  - 训练更稳定
  - 梯度流更好（残差路径无变换）
  - 现代大模型首选

### Q: RMSNorm 是什么？

RMSNorm 是 LayerNorm 的简化版，去掉了均值中心化和偏移 $\beta$：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

- 计算更高效
- LLaMA 系列使用

---

## 6. FFN 前馈网络

### Q: 标准 FFN vs SwiGLU

**标准 FFN**：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

**SwiGLU** (LLaMA/PaLM)：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3) W_2$$

其中 $\text{Swish}(x) = x \cdot \sigma(\beta x)$，GLU 门控机制提供了额外的非线性。

**SwiGLU 优势**：
- 实验表明在相同计算量下性能更好
- 门控机制让网络学会选择性激活
- 现代大模型的标准选择

### Q: FFN 中间维度为什么通常是 4d？

经验上 FFN 中间维度设为 $4 \times d_{model}$ 效果最好。使用 SwiGLU 时通常是 $\frac{8}{3} d_{model}$（参数量相当）。

---

## 7. Encoder vs Decoder

### Q: Encoder-Only / Decoder-Only / Encoder-Decoder 的区别

| 架构 | 注意力 | 代表模型 | 适用任务 |
|------|--------|----------|----------|
| Encoder-Only | 双向 | BERT, RoBERTa | 分类、NER、理解 |
| Decoder-Only | 单向（因果） | GPT, LLaMA, DeepSeek | 生成、对话 |
| Encoder-Decoder | 双向 + 因果 | T5, BART, mT5 | 翻译、摘要 |

### Q: 为什么现在的 LLM 都是 Decoder-Only？

1. **统一的 next-token prediction** 范式简单且强大
2. **Scaling law** 对 Decoder-Only 最友好
3. **GPT 系列验证了**仅 Decoder 就能做好几乎所有任务
4. **In-Context Learning** 在 Decoder 中涌现
5. **推理效率高**：KV Cache 天然适配因果注意力
6. **数据利用率**：每个 token 都是训练信号

### Q: Causal Mask（因果掩码）是什么？

在 Decoder 中，每个 token 只能看到自己和之前的 token，通过在注意力分数矩阵上加上三角掩码实现：

$$\text{Mask}_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

加在 softmax 之前：$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right)$
