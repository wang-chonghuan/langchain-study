from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

# 添加分隔函数
def print_section(section_name):
    print("\n" + "=" * 50)
    print(f"开始执行: {section_name}")
    print("=" * 50 + "\n")

# 0. 加载环境变量
print_section("加载环境变量")

load_dotenv()

# 1. 创建langgraph状态
print_section("创建langgraph状态")

'''
在代码中，通过 Annotated 的使用可以看出是否使用了 Reducer：
关键区别：
带 Reducer：使用 Annotated[类型, reducer函数] 的格式
不带 Reducer：直接声明类型，如 str, list 等
如果不使用 Reducer，当状态更新时，新值会直接覆盖旧值。而使用了 Reducer（如 add_messages），则会按照 Reducer 定义的方式来合并新旧值。
'''
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# 2. 创建第基本节点
print_section("创建基本节点")

llm = ChatOpenAI(model="gpt-4o-mini")

# 添加一个“ chatbot”节点。节点代表工作单元。它们通常是常规的 Python 函数。
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
'''
在 LangGraph 中，每个节点函数（如 `chatbot`）都遵循同一个模式：接收当前状态（State）作为输入，然后返回一个包含更新后消息的字典。当使用了 `add_messages` 这个 Reducer 时，新的 LLM 响应会被追加到状态中已存在的消息列表中，而不是覆盖它们。
'''
graph_builder.add_node("chatbot", chatbot)
#接下来，添加一个entry点。这将告诉我们的图表每次运行时从哪里开始工作。
graph_builder.add_edge(START, "chatbot")
#同样，设置一个finish点。这将指示图表“任何时候运行此节点，您都可以退出。”
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# 3. 运行graph

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


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