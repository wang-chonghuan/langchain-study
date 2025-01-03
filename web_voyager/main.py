import os
from typing import Annotated
import asyncio
from dotenv import load_dotenv
import base64

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import StateGraph, START, END
import nest_asyncio
from IPython import display
from playwright.async_api import async_playwright

# 确保在其他导入之前加载环境变量
load_dotenv()

# 导入依赖的本地模块
from graph import graph

# 添加分隔函数
async def print_section(section_name):
    print("\n" + "=" * 50)
    print(f"开始执行: {section_name}")
    print("=" * 50 + "\n")

async def call_agent(question: str, page, max_steps: int = 150):
    final_answer = None
    steps = []
    try:
        event_stream = graph.astream(
            {
                "page": page,
                "input": question,
                "scratchpad": [],
            },
            {
                "recursion_limit": max_steps,
            },
        )
        async for event in event_stream:
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            display.clear_output(wait=False)
            steps.append(f"{len(steps) + 1}. {action}: {action_input}")
            print("\n".join(steps))
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")
        raise
    finally:
        # Ensure the event stream is properly closed
        if 'event_stream' in locals():
            await event_stream.aclose()
    
    return final_answer

async def main():
    await print_section("0. 加载环境变量")
    
    # 验证环境变量是否正确加载
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 环境变量未设置")
    
    # 启动浏览器
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto("https://www.google.com")
    
    # 调用 agent
    res = await call_agent("Could you explain the WebVoyager paper (on arxiv)?", page)
    print(f"Final response: {res}")
    
    # 关闭浏览器
    await browser.close()
    
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())