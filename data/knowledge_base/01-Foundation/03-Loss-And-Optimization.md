# 损失函数与优化器

## 1. 损失函数

### Q: LLM 预训练使用什么损失函数？

**交叉熵损失（Cross-Entropy Loss）**，用于 next-token prediction：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t | x_{<t}; \theta)$$

其中：
- $T$ 是序列长度
- $x_t$ 是第 $t$ 个 token
- $P(x_t | x_{<t}; \theta)$ 是模型在给定前文下预测第 $t$ 个 token 的概率

**困惑度 (Perplexity)**：

$$\text{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t | x_{<t})\right)$$

PPL 可理解为模型平均在每个位置的"困惑"程度，越低越好。

### Q: 交叉熵、KL 散度、信息熵的关系

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

- **信息熵 $H(p)$**：分布 $p$ 的不确定性
- **KL 散度 $D_{KL}(p \| q)$**：$q$ 与 $p$ 的差异
- **交叉熵 $H(p, q)$**：用 $q$ 编码 $p$ 的平均比特数

当 $p$ 是真实标签（one-hot）时，$H(p) = 0$，因此交叉熵 = KL 散度。

---

## 2. 优化器

### Q: Adam vs AdamW 的区别 ⭐⭐⭐⭐

**Adam (Adaptive Moment Estimation)**：

| 步骤 | 公式 |
|------|------|
| 一阶矩 | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| 二阶矩 | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| 偏差修正 | $\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$ |
| 参数更新 | $\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda \theta_{t-1}$ |

**问题**：Adam 中的 L2 正则化被自适应学习率"扭曲"了，大梯度参数的正则化被减弱。

**AdamW (Decoupled Weight Decay)**：

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}$$

关键区别：权重衰减**独立于**自适应学习率。

| 特性 | Adam + L2 | AdamW |
|------|-----------|-------|
| 正则化方式 | L2 加在梯度上 | 权重衰减独立 |
| 每个参数的衰减 | 被自适应学习率缩放 | 固定 |
| 泛化效果 | 较差 | 更好 |
| LLM 训练 | 不推荐 | **推荐** |

### Q: 常用超参数设置

| 参数 | 常用值 | 说明 |
|------|--------|------|
| $\beta_1$ | 0.9 | 一阶矩衰减 |
| $\beta_2$ | 0.95~0.999 | 二阶矩衰减 |
| $\epsilon$ | 1e-8 | 数值稳定 |
| weight_decay | 0.01~0.1 | 权重衰减 |
| lr | 1e-4 ~ 3e-4 | 学习率 |

---

## 3. 学习率调度

### Q: 常用的学习率调度策略

**1. Warmup + Cosine Decay（最常用）**

$$\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}} & t \leq T_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi)) & t > T_{warmup}
\end{cases}$$

**2. WSD (Warmup-Stable-Decay)**

DeepSeek 等使用：
- Warmup 阶段线性增加
- Stable 阶段保持恒定
- Decay 阶段快速下降

### Q: 为什么需要 Warmup？

1. **训练初期参数随机**，大学习率会导致不稳定
2. Adam 的二阶矩初始估计不准确（偏差大）
3. 预热期让优化器积累可靠的统计量
4. 经验上，Warmup 步数通常设为总步数的 1%~5%

---

## 4. 梯度问题

### Q: 梯度消失和梯度爆炸如何解决？

**梯度消失：**
- 残差连接（Residual Connection）
- Pre-Norm（而非 Post-Norm）
- 合适的初始化（Xavier/He）
- 使用 ReLU 及其变体

**梯度爆炸：**
- 梯度裁剪（Gradient Clipping）：$g = \min(1, \frac{c}{\|g\|}) \cdot g$
- 权重衰减
- 较小的学习率

### Q: 混合精度训练 (Mixed Precision Training)

使用 **FP16/BF16** 进行前向和反向传播，**FP32** 维护主权重（master weights）。

| 精度 | 位数 | 范围 | 用途 |
|------|------|------|------|
| FP32 | 32 | ±3.4e38 | 主权重、优化器状态 |
| FP16 | 16 | ±6.5e4 | 前向/反向 |
| BF16 | 16 | ±3.4e38 | 前向/反向（更稳定） |

**BF16 vs FP16**：
- BF16 指数位更多（8 vs 5），范围更大
- BF16 精度更低（7 vs 10 尾数位）
- BF16 训练更稳定，不需要 loss scaling
- 现代大模型（LLaMA等）首选 BF16
