# Agent 完全指南

## 1. Agent 基础

### Q: 什么是 LLM Agent？

Agent = LLM + 规划能力 + 工具使用 + 记忆系统

**核心组件**：
```
┌─────────────────────────────────────┐
│              Agent                   │
│  ┌─────────┐  ┌──────────────────┐  │
│  │  LLM    │  │  Memory          │  │
│  │ (Brain) │  │  - Short-term    │  │
│  └────┬────┘  │  - Long-term     │  │
│       │       │  - Working       │  │
│  ┌────┴────┐  └──────────────────┘  │
│  │Planning │                         │
│  │- ReAct  │  ┌──────────────────┐  │
│  │- CoT    │  │  Tools           │  │
│  │- ToT    │  │  - Search        │  │
│  └─────────┘  │  - Code exec     │  │
│               │  - API calls     │  │
│               └──────────────────┘  │
└─────────────────────────────────────┘
```

---

## 2. ReAct 框架

### Q: ReAct 框架的原理 ⭐⭐⭐⭐

**ReAct = Reasoning + Acting**

将推理和行动交替进行：

```
问题：北京今天天气怎么样？

Thought 1: 我需要查询北京的天气信息
Action 1: search_weather(city="北京")
Observation 1: 北京今天晴，15°C

Thought 2: 我已经获得了天气信息，可以回答
Action 2: finish("北京今天天气晴朗，温度15°C")
```

**ReAct vs CoT vs Act-Only**：

| 方法 | 推理 | 行动 | 优点 |
|------|------|------|------|
| CoT | ✅ | ❌ | 推理过程透明 |
| Act-Only | ❌ | ✅ | 能用工具 |
| **ReAct** | **✅** | **✅** | **推理指导行动** |

---

## 3. Function Calling

### Q: Function Calling 的工作原理 ⭐⭐⭐⭐

**流程**：
```
用户："帮我查一下北京的天气"
    ↓
LLM 判断需要调用工具，输出结构化调用：
{
  "function": "get_weather",
  "arguments": {"city": "北京"}
}
    ↓
系统执行函数，返回结果：
{"temperature": 15, "condition": "sunny"}
    ↓
LLM 将结果整合为自然语言回复：
"北京今天天气晴朗，气温15°C"
```

**Tool 定义格式**（OpenAI 标准）：
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称"
        }
      },
      "required": ["city"]
    }
  }
}
```

---

## 4. MCP (Model Context Protocol)

### Q: MCP 协议是什么？⭐⭐⭐⭐

**MCP (Model Context Protocol)** 由 Anthropic 提出，是一个连接 LLM 和外部工具/数据源的**标准化协议**。

**架构**：
```
┌──────────┐     ┌──────────┐     ┌──────────────┐
│ LLM App  │────→│MCP Client│────→│  MCP Server   │
│(Host)    │     │          │     │  - Tools      │
│          │←────│          │←────│  - Resources  │
└──────────┘     └──────────┘     │  - Prompts    │
                                  └──────────────┘
```

**MCP 提供三种能力**：
1. **Tools**：可执行的函数/操作
2. **Resources**：可读取的数据/文件
3. **Prompts**：预定义的提示模板

### Q: MCP vs Function Calling 对比 ⭐⭐⭐⭐

| 特性 | Function Calling | MCP |
|------|-----------------|-----|
| 制定方 | OpenAI/各厂商 | Anthropic (开放标准) |
| 协议层级 | API 级别 | 应用协议 |
| 工具发现 | 静态定义 | **动态发现** |
| 生态系统 | 各自封闭 | **统一标准** |
| 传输方式 | HTTP API | stdio/SSE |
| 类比 | 各品牌充电器 | **USB-C 统一标准** |

**MCP 的优势**：
1. 一次实现，处处可用（不同 LLM 都能用）
2. 动态工具发现（不需要预定义）
3. 安全的沙箱执行
4. 社区生态共享

---

## 5. 记忆系统

### Q: Agent 的记忆系统如何设计？⭐⭐⭐⭐

**四层记忆架构**：

| 层级 | 类型 | 存储内容 | 实现方式 |
|------|------|----------|----------|
| L1 | 工作记忆 | 当前对话上下文 | Context Window |
| L2 | 短期记忆 | 近期对话摘要 | 滑动窗口 + 摘要 |
| L3 | 长期记忆 | 用户偏好/知识 | 向量数据库 |
| L4 | 外部记忆 | 文档/知识库 | RAG |

**记忆管理策略**：
1. **写入**：判断重要性 → 选择存储层级
2. **读取**：相关性检索 → 时效性过滤 → 注入上下文
3. **遗忘**：基于时效性和重要性的衰减机制
4. **整合**：定期合并相似记忆，生成高层摘要

```python
class MemorySystem:
    def __init__(self):
        self.working = []          # 当前对话
        self.short_term = deque()  # 近期摘要
        self.long_term = VectorDB() # 长期记忆

    def remember(self, info, importance):
        self.working.append(info)
        if importance > 0.7:
            self.long_term.add(info)
    
    def recall(self, query, k=5):
        relevant = self.long_term.search(query, k)
        recent = list(self.short_term)[-3:]
        return recent + relevant
```

---

## 6. 多智能体 (Multi-Agent)

### Q: 多智能体协作架构 ⭐⭐⭐

**常见模式**：

**1. 中心化（Leader-Worker）**：
```
          Orchestrator
         /     |      \
    Worker1  Worker2  Worker3
```

**2. 去中心化（Peer-to-Peer）**：
```
    Agent1 ←→ Agent2
      ↕         ↕
    Agent3 ←→ Agent4
```

**3. 层级式**：
```
         Manager
        /       \
   SubManager  SubManager
    /    \       /    \
Worker1 Worker2 Worker3 Worker4
```

### Q: 多智能体冲突解决

1. **投票机制**：多数表决
2. **辩论机制**：Agent 相互质疑和辩论
3. **仲裁者**：专门的 Judge Agent
4. **置信度加权**：按各 Agent 置信度加权

---

## 7. Agent 开发框架

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangChain | 生态丰富，Chain/Agent | 通用开发 |
| LangGraph | 状态图，复杂流程 | 多步骤 Agent |
| AutoGen | 多智能体对话 | 多 Agent 协作 |
| CrewAI | 角色扮演多智能体 | 任务分工 |
| Dify | 低代码平台 | 快速搭建 |
| Coze | 字节跳动平台 | 快速搭建 |
