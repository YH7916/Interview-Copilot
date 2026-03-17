---
name: interview-rag
description: 面试知识库 RAG 检索技能 — 通过 Query 重写、BM25+向量混合检索、RRF 融合、Rerank 重排序四阶段流水线，从面经知识库中精准召回技术概念和真实面试题。当用户询问技术概念、公司面试题、八股文时自动触发。
---

# Interview RAG — 面试知识库检索技能

## Overview

本技能为面试辅导 Agent 提供高精度知识检索能力。基于 **混合检索 + 重排序** 架构，将传统关键词匹配 (BM25) 与语义向量检索相结合，通过 Reciprocal Rank Fusion (RRF) 融合多路召回结果，最终由通义千问 Rerank 模型对候选片段重新打分，确保喂给 Agent 的参考资料高度相关。

## 核心能力

### 1. 技术概念检索 (`search_concept`)

根据技术关键词（如 "RAG"、"ReAct"、"LoRA"）从面经库中检索相关原理解析和八股文答案。

**触发场景**：用户提问涉及具体技术概念、算法原理、框架对比等。

### 2. 公司面试题检索 (`search_company_questions`)

根据公司名 + 岗位名检索历史真实面试题。

**触发场景**：用户提到特定公司或岗位的面试经验。

### 3. 混合检索流水线

每次检索自动执行四阶段流水线：

```
用户口语 Query
    ↓ ① Query Rewrite (qwen-turbo)
标准化检索词
    ↓ ② 并行召回
    ├── BM25 关键词匹配 (jieba 分词)
    └── ChromaDB 向量语义检索 (text-embedding-v3)
    ↓ ③ RRF 融合去重 (k=60)
候选文档集
    ↓ ④ Rerank 重排序 (gte-rerank)
Top-N 精选结果 → 喂给 Agent
```

## 架构

| 模块 | 文件 | 职责 |
|------|------|------|
| 向量引擎 | `engine.py` | ChromaDB 持久化、递归分块 + overlap、Embedding 入库 |
| BM25 检索 | `bm25_retriever.py` | jieba 中文分词 + BM25Okapi 关键词匹配 |
| Query 重写 | `query_rewriter.py` | 调用 qwen-turbo 将口语化查询转为检索友好格式 |
| 重排序 | `reranker.py` | 调用 DashScope gte-rerank API 对候选片段打分 |
| 混合流水线 | `hybrid_retriever.py` | 串联以上组件，RRF 融合，输出最终结果 |
| Agent 工具 | `tools.py` | 封装为 nanobot Tool，供 Agent 函数调用 |

## 使用示例

| 用户输入 | 触发工具 | 检索行为 |
|----------|----------|----------|
| "RAG 的原理是什么？" | `search_concept` | 检索 RAG 相关技术文档 |
| "查查字节的面经" | `search_company_questions` | Query 重写为 "字节跳动 面试经验 面试题"，检索真实面试题 |
| "LoRA 和 QLoRA 有什么区别？" | `search_concept` | 检索微调相关对比分析 |
