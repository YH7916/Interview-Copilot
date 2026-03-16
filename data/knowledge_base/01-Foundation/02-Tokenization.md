# 分词算法 (Tokenization)

## 1. BPE (Byte Pair Encoding)

### Q: BPE 分词算法的原理 ⭐⭐⭐

**BPE（字节对编码）** 是一种子词分词算法，从字符级别开始，迭代合并最频繁的字符对。

**训练过程：**

1. 初始化词表为所有字符
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对，加入词表
4. 重复步骤 2-3，直到达到目标词表大小

**示例：**

```
语料: "low lower lowest"

初始: l o w </w> | l o w e r </w> | l o w e s t </w>

Step 1: 最频繁对 (l, o) → "lo"
Step 2: 最频繁对 (lo, w) → "low"  
Step 3: 最频繁对 (low, e) → "lowe"
...
```

**编码过程**：对新文本，贪心地匹配词表中最长的子词。

### Q: BPE vs WordPiece vs SentencePiece

| 特性 | BPE | WordPiece | SentencePiece |
|------|-----|-----------|---------------|
| 合并策略 | 最高频率对 | 最大似然增益 | BPE/Unigram 在句子片段上 |
| 预分词 | 需要 | 需要 | 不需要（直接处理原始文本） |
| 空格处理 | 保留 | `##` 标记续接 | `▁` 标记开头 |
| 使用模型 | GPT, LLaMA | BERT | T5, LLaMA, ChatGLM |

### Q: WordPiece 与 BPE 的核心区别

WordPiece 不是简单地合并最频繁的对，而是选择**使语言模型似然增加最大**的对。

合并标准：

$$\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$$

这实际上是互信息（Mutual Information），倾向于合并那些频繁共现但单独出现不太频繁的对。

---

## 2. Unigram 模型

### Q: Unigram 分词与 BPE 的区别

| 特性 | BPE | Unigram |
|------|-----|---------|
| 方向 | 自底向上（合并） | 自顶向下（裁剪） |
| 初始化 | 字符 | 大词表 |
| 过程 | 逐步合并 | 逐步删除 |
| 分词 | 确定性 | 概率性（可采样） |

**Unigram 模型**：
- 初始化一个很大的候选词表
- 为每个子词分配概率：$P(x) = \prod_{i=1}^{n} P(x_i)$
- 用 Viterbi 算法找最优分割
- 迭代删除使似然下降最小的子词

---

## 3. 中文分词特殊问题

### Q: 中文 LLM 的分词有什么特殊之处？

**挑战**：
- 中文没有天然的空格分隔
- 词粒度不确定：「中华人民共和国」是一个词还是多个词？
- 字符集远大于英文

**解决方案**：
1. **字级别**：每个汉字作为一个 token（BERT-Chinese）
2. **SentencePiece**：直接在字节/字符序列上训练 BPE/Unigram
3. **混合方案**：先中文分词（jieba等），再 BPE

**现代做法**（LLaMA/ChatGLM）：
- 使用 SentencePiece 的 BPE 模式
- 对中文，通常会在预训练语料中增大中文比例
- 扩展 LLaMA 的词表加入中文 token（如 Chinese-LLaMA-Alpaca）

### Q: 词表大小如何选择？

| 词表大小 | 优点 | 缺点 |
|----------|------|------|
| 小（~8K） | Embedding 参数少 | 序列长、压缩率低 |
| 中（32K~64K） | 平衡 | 常见选择 |
| 大（100K+） | 序列短、压缩率高 | Embedding 参数大 |

常见词表大小：
- GPT-2: 50,257
- LLaMA: 32,000
- LLaMA-2: 32,000 
- LLaMA-3: 128,256
- ChatGLM: 65,024
- DeepSeek-V3: 128,000

### Q: token 数量与序列长度的关系？

- 英文：约 1 token ≈ 0.75 个单词 ≈ 4 个字符
- 中文：约 1 token ≈ 1-2 个汉字（取决于词表）
- 代码：通常比自然语言需要更多 token

**压缩率**影响：
- 较高的压缩率 → 较短的序列 → 更快的推理
- 但过高的压缩率可能损失细粒度信息
