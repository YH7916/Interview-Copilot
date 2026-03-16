# 对齐技术 (Alignment)

## 1. RLHF (Reinforcement Learning from Human Feedback)

### Q: RLHF 的完整流程 ⭐⭐⭐⭐

**三个阶段**：

**阶段 1：SFT (Supervised Fine-Tuning)**
- 用高质量人工标注数据做有监督微调
- 让模型学会遵循指令

**阶段 2：Reward Model Training**
- 人工对模型回答进行排序
- 训练奖励模型 $r(x, y)$ 学习人类偏好
- 损失函数：

$$\mathcal{L}_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l))$$

**阶段 3：PPO (Proximal Policy Optimization)**
- 用奖励模型指导策略优化

---

## 2. PPO 详解

### Q: PPO 的四个组件 ⭐⭐⭐⭐

| 组件 | 作用 | 描述 |
|------|------|------|
| Actor (策略模型) | 生成回答 | 正在训练的 LLM |
| Critic (价值模型) | 评估状态价值 | $V(s)$ 估计 |
| Reward Model | 评分 | 评估回答质量 |
| Reference Model | 防止偏离 | 冻结的 SFT 模型 |

**PPO 目标函数**：

$$\mathcal{L}^{PPO} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\theta_{old}}} A_t, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1-\epsilon, 1+\epsilon\right) A_t\right)\right]$$

**总奖励**：

$$R(x,y) = r_\phi(x,y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

KL 惩罚防止策略偏离参考模型太远。

---

## 3. DPO (Direct Preference Optimization)

### Q: DPO 的数学推导 ⭐⭐⭐

**核心思想**：跳过奖励模型训练，直接从偏好数据优化策略。

**推导过程**：

从 RLHF 的最优策略出发：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

反解奖励函数：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 模型：

$$P(y_w \succ y_l) = \sigma(r(x,y_w) - r(x,y_l))$$

得到 **DPO 损失**：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### Q: DPO vs PPO 对比

| 特性 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高（4个模型） | 低（2个模型） |
| 超参数 | 多 | 少 |
| 在线/离线 | 在线 | 离线 |
| 稳定性 | 较差 | 较好 |
| 效果 | 某些场景更好 | 简单有效 |

---

## 4. GRPO (Group Relative Policy Optimization)

### Q: GRPO 的原理（DeepSeek 核心算法）⭐⭐⭐⭐⭐

**核心创新**：去掉 Critic 模型，用**组内相对奖励**估算优势函数。

**算法流程**：
1. 对每个问题 $x$，从策略 $\pi_\theta$ 采样一组回答 $\{y_1, ..., y_G\}$
2. 用奖励模型 $r$ 对每个回答打分 $\{r_1, ..., r_G\}$
3. 计算组内标准化的优势：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,...,r_G\})}{\text{std}(\{r_1,...,r_G\})}$$

4. 优化目标：

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}\hat{A}_i, \text{clip}(\cdot)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right)\right]$$

### Q: GRPO vs PPO 的关键区别

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic 模型 | 需要（参数量=Actor） | **不需要** |
| 优势估计 | GAE (需要 Critic) | 组内相对奖励 |
| 显存占用 | 大（4个模型） | 小（2个模型+采样） |
| 训练效率 | 较低 | 更高 |
| 使用模型 | InstructGPT | **DeepSeek-R1** |

---

## 5. 其他对齐方法

### DAPO (Decoupled Alignment from Direct Preference Optimization)
- 解耦对齐的不同方面
- 分别处理有用性和安全性

### REINFORCE++
- REINFORCE 的改进版
- 加入 baseline 减小方差
- 比 PPO 简单，比 REINFORCE 稳定

### Q: 对齐方法时间线

```
RLHF/PPO (2022, InstructGPT)
    ↓
DPO (2023, Stanford)
    ↓
KTO, IPO, ORPO (2024, 各种改进)
    ↓
GRPO (2025, DeepSeek-R1)
    ↓
DAPO, REINFORCE++ (2025-2026)
```

---

## 6. 奖励函数设计

### Q: 如何为 Function Calling 场景设计奖励函数？

```python
def reward_function_calling(response, ground_truth):
    reward = 0.0
    
    # 1. 格式正确性 (0.3)
    if is_valid_json(response.tool_calls):
        reward += 0.3
    
    # 2. 工具选择正确性 (0.3)
    if response.tool_name == ground_truth.tool_name:
        reward += 0.3
    
    # 3. 参数正确性 (0.3)
    param_score = compute_param_overlap(
        response.params, ground_truth.params
    )
    reward += 0.3 * param_score
    
    # 4. 执行结果 (0.1)
    if execute(response) == ground_truth.result:
        reward += 0.1
    
    return reward
```
