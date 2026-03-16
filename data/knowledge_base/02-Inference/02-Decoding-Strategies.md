# 解码策略

## 1. 贪心解码 (Greedy Decoding)

每一步选择概率最大的 token：

$$x_t = \arg\max_x P(x | x_{<t})$$

**优点**：快速、确定性
**缺点**：无法探索，容易重复，非全局最优

---

## 2. Beam Search

维护 $b$ 个最优候选序列：

- 每步扩展所有候选，保留前 $b$ 个得分最高的
- 得分通常用 log 概率之和（可加长度惩罚）

$$\text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t | y_{<t})$$

---

## 3. 采样策略

### Temperature Sampling

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$：更尖锐（确定性更强）
- $T = 1$：原始分布
- $T > 1$：更平坦（更随机）

### Top-K Sampling

只从概率最大的 K 个 token 中采样。

### Top-P (Nucleus) Sampling

从累积概率达到 $p$ 的最小 token 集合中采样：

$$\sum_{x \in V_p} P(x) \geq p$$

### Q: Top-K vs Top-P 的区别？

| 特性 | Top-K | Top-P |
|------|-------|-------|
| 候选集大小 | 固定 K | 动态（取决于分布） |
| 分布尖锐时 | 可能包含低概率词 | 自动缩小 |
| 分布平坦时 | 可能排除合理词 | 自动扩大 |
| 推荐 | 简单场景 | 更灵活 |

---

## 4. 重复惩罚

$$P'(x_i) = \begin{cases} P(x_i) / \alpha & \text{if } x_i \in \text{generated} \\ P(x_i) & \text{otherwise} \end{cases}$$

常用方法：
- Repetition Penalty：已生成 token 概率除以 $\alpha$
- Frequency Penalty：按出现次数线性惩罚
- Presence Penalty：出现过就惩罚固定值
