# 字节跳动 LLM/Agent 真实面试题

> 以下面试题整理自 2025-2026 年牛客网真实面经

---

## 面经 1：字节跳动 - 豆包大模型团队（2026.01）

**一面（基础+项目）**：
1. Self-Attention 的计算过程和复杂度分析
2. 为什么要除以 √d_k？
3. Multi-Head Attention 的作用，为什么不用单头？
4. KV Cache 的原理，如何计算显存占用？
5. GRPO 和 PPO 的区别，为什么 DeepSeek 用 GRPO？
6. 手写 Self-Attention（Python）
7. 项目中 RAG 的检索效果如何优化？

**二面（深入+系统设计）**：
1. MLA 的原理，与 GQA 相比的优势？
2. Flash Attention 如何实现的？为什么更快？
3. DPO 的数学推导
4. 如何设计一个 Agent 的记忆系统？
5. MCP 和 Function Calling 的区别
6. 设计一个支持多轮对话的大模型服务

---

## 面经 2：字节跳动 - AI 应用研发（2026.01）

1. Transformer 的整体架构，Encoder 和 Decoder 的区别
2. RoPE 旋转位置编码的原理
3. LoRA 的原理，秩 r 怎么选？
4. RLHF 三个阶段分别是什么？
5. PPO 中 Critic 模型的作用
6. GRPO 怎么去掉 Critic 的？优势是什么？
7. RAG 中 embedding 模型怎么选？chunk 大小怎么定？
8. BM25 和向量检索的区别
9. 手写 Multi-Head Attention
10. LRU Cache（LeetCode 146）

---

## 面经 3：字节跳动 - 搜索与推荐（2025.12）

1. BPE 分词算法的训练和推理过程
2. LayerNorm vs BatchNorm
3. Pre-Norm vs Post-Norm，为什么现在都用 Pre-Norm？
4. Adam 和 AdamW 的区别
5. 混合精度训练，BF16 vs FP16 的区别
6. KV Cache 优化方法有哪些？
7. DeepSeek-V3 的 MoE 架构设计
8. 模型量化的方法，GPTQ 和 AWQ 的区别
9. 投机解码（Speculative Decoding）的原理
10. 手写 BPE 分词器

---

## 面经 4：字节跳动 - Agent 平台（2025.12）

1. 什么是 LLM Agent？核心组件有哪些？
2. ReAct 框架的原理
3. Function Calling 的工作流程
4. MCP 协议的架构和通信方式
5. Agent 如何处理多步骤任务的错误恢复？
6. 多智能体系统的协作模式
7. 如何评估 Agent 的准确性和可靠性？
8. 设计一个带记忆的多轮对话 Agent
9. 手写余弦相似度搜索函数
10. TopK 问题（堆排序）

---

## 面经 5：字节跳动 - 大模型算法（2025.11）

1. Transformer 的 FLOPs 如何估算？
2. GQA 的原理和实现
3. SwiGLU 激活函数
4. RMSNorm 是什么？和 LayerNorm 的区别
5. 长上下文怎么做？NTK-aware RoPE 的原理
6. DPO 和 RLHF 训练哪个更好？各自优缺点
7. GRPO 的优势函数是怎么计算的？
8. 大模型幻觉的原因和解决方案
9. 手写 LoRA 层
10. 合并 K 个有序链表

---

## 面经 6：字节跳动 - 国际化 TikTok（2025.11）

1. Self-Attention 为什么用 Softmax？能换成别的吗？
2. Causal Mask 是什么？为什么需要？
3. Tokenizer 对模型性能的影响
4. SFT 数据质量 vs 数量哪个更重要？
5. PPO 中 KL 惩罚的作用
6. GRPO 中 group 大小怎么选？
7. PagedAttention (vLLM) 的原理
8. Continuous Batching 是什么？
9. 手写 Beam Search 解码
10. 二叉树层序遍历

---

## 面经 7：字节跳动 - 飞书智能助手（2025.10）

1. 完整描述 RAG 系统的架构
2. 向量数据库的选型（Milvus/Weaviate/Chroma）
3. 查询改写的方法（HyDE、Multi-Query）
4. Reranking 的作用和常用模型
5. 如何评估 RAG 系统？RAGAS 指标
6. Agent 的 Planning 模块如何实现？
7. 手写 KV Cache Attention
8. LRU Cache 设计

---

## 面经 8：字节跳动 - AI Infra（2025.10）

1. 大模型分布式训练的并行策略
2. DeepSpeed ZeRO 三个阶段的区别
3. 张量并行和流水线并行怎么选？
4. All-Reduce 和 Ring All-Reduce 的区别
5. 梯度累积的作用和实现
6. vLLM 的架构和核心优化技术
7. CUDA kernel 优化的基本方法
8. 如何设计一个大模型推理服务？

---

## 面经 9：字节跳动 - 安全与治理（2026.02）

1. LLM 常见的安全风险
2. 越狱攻击（Jailbreak）的类型和防御
3. 提示注入（Prompt Injection）如何防范？
4. RLHF 对齐训练如何增强安全性？
5. Red Teaming 的流程
6. 大模型的偏见检测和缓解

---

## 面经 10：字节跳动 - 推理优化（2026.01）

1. KV Cache 显存计算公式（MHA/GQA/MLA 分别算）
2. Flash Attention 的 tiling 策略和 online softmax
3. 投机解码为什么能保证输出分布一致？
4. 模型蒸馏和量化的区别
5. INT4 量化后精度下降怎么办？
6. 如何从系统层面优化大模型推理延迟？

---

## 面经 11：字节跳动 - 教育 AI（2026.02）

1. 为什么现在的 LLM 都是 Decoder-Only？
2. Encoder-Decoder 架构适合什么任务？
3. test-time compute 是什么？
4. DeepSeek-R1 的训练方法
5. 多模态大模型（VLM）的架构
6. 如何将 LLM 应用于教育场景？个性化学习路径设计

---

## 📊 频率统计

| 题目 | 出现次数 | 频率 |
|------|----------|------|
| GRPO vs PPO | 7/11 | 63.6% |
| Self-Attention 计算 | 6/11 | 54.5% |
| KV Cache | 6/11 | 54.5% |
| 手写 Attention | 5/11 | 45.5% |
| LoRA | 5/11 | 45.5% |
| DPO 推导 | 4/11 | 36.4% |
| Flash Attention | 4/11 | 36.4% |
| RAG 优化 | 4/11 | 36.4% |
| MCP/Agent | 4/11 | 36.4% |
| ReAct | 3/11 | 27.3% |
| BPE 分词 | 3/11 | 27.3% |
| MoE | 3/11 | 27.3% |
| 系统设计题 | 3/11 | 27.3% |
