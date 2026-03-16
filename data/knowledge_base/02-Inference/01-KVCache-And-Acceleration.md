# KV Cache 与推理加速

## 1. KV Cache

### Q: KV Cache 的原理 ⭐⭐⭐⭐⭐

在自回归生成中，每生成一个新 token，需要对所有之前的 token 计算 K 和 V。KV Cache 将已计算的 K、V 缓存起来，避免重复计算。

**无 KV Cache**：
```
生成 token 5 时：计算 token 1-4 的 K,V → 计算 token 5 的注意力
生成 token 6 时：重新计算 token 1-5 的 K,V → 计算 token 6 的注意力
```

**有 KV Cache**：
```
生成 token 5 时：读取缓存的 K1-4, V1-4 → 只计算 K5, V5 → 拼接 → 注意力
生成 token 6 时：读取缓存的 K1-5, V1-5 → 只计算 K6, V6 → 拼接 → 注意力
```

### Q: KV Cache 的显存计算 ⭐⭐⭐⭐⭐

**公式**：

$$\text{KV Cache Memory} = 2 \times b \times n_{kv} \times L \times s \times d_{head} \times \text{sizeof(dtype)}$$

其中：
- 2 = K 和 V
- $b$ = batch size
- $n_{kv}$ = KV 头数
- $L$ = 层数
- $s$ = 序列长度
- $d_{head}$ = 每个头的维度
- sizeof(dtype) = 2 (FP16) 或 1 (INT8)

**示例：LLaMA-2 7B**
- L=32, n_kv=32, d_head=128, s=4096, b=1, FP16
- KV Cache = 2 × 1 × 32 × 32 × 4096 × 128 × 2 bytes = **2 GB**

### Q: 如何减少 KV Cache？

| 方法 | 原理 | 减少比例 |
|------|------|----------|
| MQA | 所有 Q 头共享 1 个 KV 头 | 1/h |
| GQA | 每组 Q 头共享 1 个 KV 头 | g/h |
| MLA | 压缩到低维潜在空间 | ~6.7% |
| 量化 | KV Cache 用 INT8/INT4 | 50%/75% |
| 窗口注意力 | 只保留最近的 KV | window/total |

---

## 2. Flash Attention

### Q: Flash Attention 的原理 ⭐⭐⭐⭐

**核心问题**：标准 Attention 需要存储 $n \times n$ 的注意力矩阵，显存 $O(n^2)$。

**核心思想**：利用 GPU 内存层次（SRAM vs HBM），采用 **tiling（分块）** 技术：
1. 将 Q, K, V 分成小块
2. 在 SRAM 中计算局部注意力
3. 使用 online softmax 技巧合并结果
4. 避免在 HBM 中存储完整的 $n \times n$ 矩阵

**GPU 内存层次**：
```
SRAM:   ~20MB,  ~19TB/s  (On-chip, 快)
HBM:    ~40GB,  ~1.5TB/s (Off-chip, 慢)
```

**FlashAttention 效果**：
- 显存：$O(n^2) \to O(n)$
- 速度：2-4x 加速
- 精确计算（非近似）

---

## 3. PagedAttention (vLLM)

### Q: PagedAttention 解决什么问题？

**问题**：KV Cache 预分配导致显存浪费（因为序列长度不确定）

**方案**：借鉴操作系统的虚拟内存/分页机制：
- KV Cache 存储在非连续的物理块中
- 通过页表（Block Table）管理映射
- 按需分配，消除内部碎片
- 支持 KV Cache 在请求间**共享**（如 beam search）

**效果**：显存利用率接近 100%，吞吐量提升 2-4x。

---

## 4. Continuous Batching

### Q: 静态 Batching vs 连续 Batching

| 特性 | 静态 Batching | Continuous Batching |
|------|---------------|---------------------|
| 批次管理 | 等最长序列结束 | 动态插入/移除请求 |
| GPU 利用率 | 低（padding） | 高 |
| 延迟 | 高 | 低 |
| 吞吐 | 低 | 高（2-3x） |

---

## 5. Speculative Decoding（投机解码）

### Q: 投机解码的原理 ⭐⭐⭐

**核心思想**：用小型 draft model 快速生成多个候选 token，然后用大模型并行验证。

**流程**：
1. Draft model 自回归生成 $\gamma$ 个 token
2. Target model **一次前向传播**验证这 $\gamma$ 个 token
3. 从第一个不匹配的位置开始 reject，保留匹配的
4. 从不匹配位置重新采样一个正确的 token

**关键性质**：生成结果的分布与仅用大模型生成**完全一致**。

**加速比**：与 $\alpha^\gamma$ 相关，其中 $\alpha$ 是 draft model 与 target model 的 acceptance rate。

---

## 6. 推理框架对比

| 框架 | 核心技术 | 特点 |
|------|----------|------|
| vLLM | PagedAttention | 高吞吐，显存效率 |
| TensorRT-LLM | 图优化+量化 | NVIDIA 官方，极致性能 |
| SGLang | RadixAttention | 前缀缓存，编程框架 |
| DeepSpeed-Inference | 张量并行+量化 | 分布式推理 |
| Ollama | GGUF 格式 | 本地部署，易用 |
