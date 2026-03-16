import asyncio
import sys
import os

# Ensure the nanobot module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nanobot.skills.interview_rag.tools import SearchConceptTool, SearchCompanyQuestionsTool

async def main():
    print("Testing SearchConceptTool...")
    concept_tool = SearchConceptTool()
    
    # Test valid case
    res1 = await concept_tool.execute("JVM垃圾回收机制")
    print(f"Valid Topic Result: {res1}")
    
    # Test error fallback
    res2 = await concept_tool.execute("simulate_error")
    print(f"Error Topic Result: {res2}")
    
    print("\nTesting SearchCompanyQuestionsTool...")
    company_tool = SearchCompanyQuestionsTool()
    
    # Test valid case
    res3 = await company_tool.execute("字节跳动", "后端开发")
    print(f"Valid Company Result: {res3}")
    
    # Test error fallback
    res4 = await company_tool.execute("simulate_error", "前端开发")
    print(f"Error Company Result: {res4}")

if __name__ == "__main__":
    asyncio.run(main())
