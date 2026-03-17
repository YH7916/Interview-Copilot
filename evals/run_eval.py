"""
LLMOps 自动化评测流水线 (LLM-as-a-Judge)
架构：
  1. copilot.rag 检索相关知识 (RAG)
  2. Qwen 基于检索结果生成回答 (Generator)
  3. Qwen 作为严苛裁判对回答打分 (Judge)
"""
import asyncio
import json
import sys
from pathlib import Path

from copilot.rag.hybrid_retriever import HybridRetriever

# 确保项目根目录在 Python PATH 内，copilot/ 包才能正常导入
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import dashscope
from dotenv import load_dotenv
from copilot.config import get_dashscope_api_key
from copilot.memory.weakness_tracker import WeaknessTracker

load_dotenv(BASE_DIR / ".env")

# ==========================================
# 配置：从 .env 或 ~/.nanobot/config.json 读取 API Key
# ==========================================
dashscope.api_key = get_dashscope_api_key()
if not dashscope.api_key:
    print("⚠️ 未找到 DASHSCOPE_API_KEY，请在 .env 或 ~/.nanobot/config.json 中配置。")

DATASET_PATH = BASE_DIR / "evals" / "datasets" / "golden_set.json"


# ==========================================
# 亮点 1: RAG Pipeline — 用 copilot.rag 检索 + Qwen 生成答案
# 直接复用 copilot 层的 RAG 引擎，不造轮子，不污染 nanobot
# ==========================================
def ask_agent_with_rag(question: str) -> tuple[str, list[dict]]:
    """用 copilot RAG 检索上下文，再用 Qwen 生成专业回答"""
    results: list[dict] = []
    try:
        retriever = HybridRetriever()
        results = asyncio.run(retriever.search(question, top_k_retrieve=5, top_n_rerank=2))
        docs = [r.get("text", "") for r in results]
        context = "\n\n".join(docs) if docs else "（未检索到相关资料）"
    except Exception as e:
        context = f"（RAG 检索失败: {e}）"

    prompt = f"""你是一个 AI 面试辅导 Agent。请基于以下参考资料，简明准确地回答面试问题。
如果资料不足，也可以结合自身知识补充，但不要编造不存在的概念。

【参考资料】
{context}

【面试问题】
{question}

请直接给出回答："""

    try:
        response = dashscope.Generation.call(
            model="qwen-max", prompt=prompt, result_format="text"
        )
        return response.output.text.strip(), results
    except Exception as e:
        return f"[生成回答失败: {e}]", []


# ==========================================
# 亮点 2: LLM-as-a-Judge — 结构化双维度打分
# ==========================================
JUDGE_PROMPT = """
你是一个极其严苛的高级 AI 面试官。请对候选人 AI Agent 的回答质量进行评估。

【评估标准】（1-5分）
1. 准确性 (Accuracy)：是否包含标准答案的核心知识点？是否有技术事实错误？
2. 幻觉控制 (Faithfulness)：是否过度发散或编造了错误概念？

【数据】
- 问题：{question}
- 标准答案：{expected_answer}
- Agent 回答：{actual_answer}

【输出要求】严格输出合法 JSON，无 Markdown：
{{"accuracy_score": <1-5>, "faithfulness_score": <1-5>, "reason": "<简要点评>"}}
"""

def evaluate_answer(question: str, expected: str, actual: str) -> dict:
    """调用 Qwen 作为裁判打分"""
    prompt = JUDGE_PROMPT.format(
        question=question, expected_answer=expected, actual_answer=actual
    )
    try:
        response = dashscope.Generation.call(
            model="qwen-max", prompt=prompt, result_format="text"
        )
        text = response.output.text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"⚠️ 裁判打分失败: {e}")
        return {"accuracy_score": 0, "faithfulness_score": 0, "reason": "裁判系统异常"}


# ==========================================
# 亮点 3: CI/CD 评测流水线 — RAG Agent vs. Judge
# ==========================================
def run_evaluation_pipeline():
    print("🚀 启动 LLMOps 自动化评测流水线 (RAG + LLM-as-a-Judge)...")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    total_accuracy = total_faithfulness = 0
    total_recall = 0.0
    recall_count = 0

    print("⏳ 正在让 Agent 作答并交由 AI 裁判打分，请稍候...\n")

    eval_records: list[dict] = []

    for i, data in enumerate(dataset, 1):
        question, expected = data["question"], data["expected_answer"]

        print(f"[{i}/{len(dataset)}] 评测中: {question}")
        agent_answer, rag_results = ask_agent_with_rag(question)
        retrieved_sources = {r.get("metadata", {}).get("source", "") for r in rag_results}
        expected_sources = set(data.get("expected_sources", []))
        recall = len(retrieved_sources & expected_sources) / len(expected_sources) if expected_sources else None
        if recall is not None:
            total_recall += recall
            recall_count += 1

        preview = agent_answer[:80] + "..." if len(agent_answer) > 80 else agent_answer
        print(f"  📝 Agent: {preview}")
        if recall is not None:
            print(f"  🔍 Recall@K: {recall:.2f}")

        result = evaluate_answer(question, expected, agent_answer)
        print(f"  👉 准确性: {result.get('accuracy_score')}/5  幻觉控制: {result.get('faithfulness_score')}/5")
        print(f"  🗣️  点评: {result.get('reason')}\n")

        total_accuracy += result.get("accuracy_score", 0)
        total_faithfulness += result.get("faithfulness_score", 0)
        eval_records.append({
            "question": question,
            "expected": expected,
            "actual": agent_answer,
            "accuracy_score": result.get("accuracy_score", 0),
            "faithfulness_score": result.get("faithfulness_score", 0),
            "reason": result.get("reason", ""),
            "recall_at_k": recall,
            "retrieved_sources": sorted(s for s in retrieved_sources if s),
            "expected_sources": sorted(expected_sources),
            "rag_results": rag_results,
        })

    n = len(dataset)
    print("=" * 40)
    print("📊 评测报告 (Metrics Summary)")
    print(f"🎯 平均准确性  (Accuracy):    {total_accuracy/n:.1f} / 5.0")
    print(f"🛡️  平均无幻觉率 (Faithfulness): {total_faithfulness/n:.1f} / 5.0")
    if recall_count > 0:
        print(f"🔍 平均检索召回率 (Recall@K): {total_recall/recall_count:.2f}")
    print("=" * 40)

    # ==========================================
    # 亮点 4: 自动更新长期记忆 — 错题本
    # 面试可吹: 系统有自学习能力，每次跑完评测自动沉淀薄弱点
    # ==========================================
    print("\n📝 正在更新错题本 (weakness_log.md)...")
    tracker = WeaknessTracker()
    tracker.update(eval_records)
    print(f"✅ 错题本已更新: {tracker.log_path}")


if __name__ == "__main__":
    run_evaluation_pipeline()
