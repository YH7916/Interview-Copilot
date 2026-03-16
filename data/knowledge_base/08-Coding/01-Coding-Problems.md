# 手撕代码

## 1. Self-Attention ⭐⭐⭐⭐⭐

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model
        
        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        Q = self.W_Q(x)  # (batch, seq_len, d_k)
        K = self.W_K(x)  # (batch, seq_len, d_k)
        V = self.W_V(x)  # (batch, seq_len, d_v)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
```

---

## 2. Multi-Head Attention ⭐⭐⭐⭐⭐

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换并拆分成多头
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # shape: (batch, n_heads, seq_len, d_k)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_O(context)
        return output
```

---

## 3. Transformer FLOPs 估算

```python
def estimate_transformer_flops(
    vocab_size, d_model, n_layers, n_heads, seq_len, batch_size
):
    """估算 Transformer 一次前向传播的 FLOPs"""
    
    # 每个 Attention 层
    # QKV 投影: 3 * 2 * batch * seq * d_model^2
    qkv_flops = 3 * 2 * batch_size * seq_len * d_model ** 2
    # Attention 矩阵: 2 * batch * n_heads * seq^2 * d_k
    d_k = d_model // n_heads
    attn_flops = 2 * batch_size * n_heads * seq_len ** 2 * d_k
    # 输出投影: 2 * batch * seq * d_model^2
    out_flops = 2 * batch_size * seq_len * d_model ** 2
    
    # FFN: 2 * 2 * batch * seq * d_model * 4 * d_model
    ffn_flops = 2 * 2 * batch_size * seq_len * d_model * 4 * d_model
    
    # 总 FLOPs
    per_layer = qkv_flops + attn_flops + out_flops + ffn_flops
    total = per_layer * n_layers
    
    # Embedding + LM Head
    embed_flops = 2 * 2 * batch_size * seq_len * vocab_size * d_model
    
    return total + embed_flops

# 示例：LLaMA-7B
flops = estimate_transformer_flops(
    vocab_size=32000, d_model=4096, n_layers=32,
    n_heads=32, seq_len=2048, batch_size=1
)
print(f"FLOPs: {flops / 1e12:.1f} TFLOPs")
```

---

## 4. BPE 分词器

```python
from collections import Counter

class SimpleBPE:
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = set()
    
    def train(self, corpus):
        # 初始化：字符级别
        words = []
        for word in corpus.split():
            words.append(list(word) + ['</w>'])
            self.vocab.update(word)
        self.vocab.add('</w>')
        
        while len(self.vocab) < self.vocab_size:
            # 统计相邻对频率
            pairs = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += 1
            
            if not pairs:
                break
            
            # 找最频繁的对
            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)
            
            # 合并
            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words
    
    def tokenize(self, text):
        tokens = list(text) + ['</w>']
        for a, b in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(a + b)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
```

---

## 5. Beam Search

```python
def beam_search(model, input_ids, beam_width=3, max_len=50, eos_token_id=2):
    # 初始化：每个 beam 是 (序列, 累积 log 概率)
    beams = [(input_ids, 0.0)]
    completed = []
    
    for _ in range(max_len):
        all_candidates = []
        
        for seq, score in beams:
            if seq[-1] == eos_token_id:
                completed.append((seq, score))
                continue
            
            # 获取下一个 token 的概率
            logits = model(seq)  # (vocab_size,)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 取 top-k 候选
            topk_probs, topk_ids = log_probs.topk(beam_width)
            
            for prob, token_id in zip(topk_probs, topk_ids):
                new_seq = seq + [token_id.item()]
                new_score = score + prob.item()
                all_candidates.append((new_seq, new_score))
        
        if not all_candidates:
            break
        
        # 保留 top beam_width 个
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:beam_width]
    
    # 加入未完成的
    completed.extend(beams)
    
    # 长度归一化
    def length_normalize(seq_score, alpha=0.6):
        seq, score = seq_score
        return score / (len(seq) ** alpha)
    
    completed.sort(key=length_normalize, reverse=True)
    return completed[0][0]
```

---

## 6. KV Cache Attention

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        # KV Cache
        self.k_cache = None
        self.v_cache = None
    
    def forward(self, x, use_cache=True):
        B, L, _ = x.shape
        
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        if use_cache and self.k_cache is not None:
            K = torch.cat([self.k_cache, K], dim=2)
            V = torch.cat([self.v_cache, V], dim=2)
        
        if use_cache:
            self.k_cache = K.detach()
            self.v_cache = V.detach()
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_O(out)
    
    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
```

---

## 7. LoRA 实现

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features), requires_grad=False
        )
        
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # 原始路径 + LoRA 路径
        base_output = F.linear(x, self.weight)
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base_output + self.scaling * lora_output
```

---

## 8. 余弦相似度检索

```python
import numpy as np

def cosine_similarity_search(query_vec, doc_vecs, top_k=5):
    """
    query_vec: (d,)
    doc_vecs: (n, d)
    """
    # 归一化
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    
    # 余弦相似度
    similarities = np.dot(doc_norms, query_norm)
    
    # Top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(idx, similarities[idx]) for idx in top_indices]
```

---

## 9. RoPE 旋转位置编码

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq, xk: (batch, seq_len, n_heads, d_head)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = freqs_cis[:xq.shape[1]]
    freqs_cis = freqs_cis[None, :, None, :]  # broadcast
    
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

---

## 10. LeetCode 高频算法题

面试中还会考基础算法，以下是 LLM 岗位高频题：

| 题目 | 难度 | 频率 |
|------|------|------|
| LRU Cache | Medium | ⭐⭐⭐⭐⭐ |
| TopK 问题 | Medium | ⭐⭐⭐⭐ |
| 合并 K 个有序链表 | Hard | ⭐⭐⭐ |
| 二叉树层序遍历 | Medium | ⭐⭐⭐ |
| 最长公共子序列 | Medium | ⭐⭐⭐ |
| 岛屿数量 | Medium | ⭐⭐⭐ |
| 快速排序 / 归并排序 | Medium | ⭐⭐⭐ |
| 滑动窗口最大值 | Hard | ⭐⭐⭐ |
