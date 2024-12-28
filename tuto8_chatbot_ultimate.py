from typing import Annotated

from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import StateGraph, START, END


# 添加分隔函数
def print_section(section_name):
    print("\n" + "=" * 50)
    print(f"开始执行: {section_name}")
    print("=" * 50 + "\n")


print_section("0. 加载环境变量")

load_dotenv()

memory = MemorySaver()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):
    # 第一次执行,直到中断点
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        config, 
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            print("Assistant:", event["messages"][-1].content)
    
    # 获取当前状态,检查是否需要继续执行
    snapshot = graph.get_state(config)
    if snapshot.next == ("tools",):
        print("\n[系统] 准备执行搜索,是否继续? (y/n)")
        if input().lower() == 'y':
            # 继续执行
            events = graph.stream(None, config, stream_mode="values")
            for event in events:
                if "messages" in event:
                    print("Assistant:", event["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
