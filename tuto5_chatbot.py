import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# 添加分隔函数
def print_section(section_name):
    print("\n" + "="*50)
    print(f"开始执行: {section_name}")
    print("="*50 + "\n")

# 0. 加载环境变量
print_section("加载环境变量")

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

model = ChatOpenAI(model="gpt-4o-mini")

response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

print(response)

# 1. 定义一个状态图
print_section("定义一个状态图")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# Async function for node:
'''
由于代码中使用了 await 关键字进行异步操作，app.ainvoke需要在异步函数（async函数）内部运行，并且要通过 asyncio 来执行，否则会报错。简单来说就是需要把代码包装在 async def main() 函数中，然后用 asyncio.run(main()) 来运行。
'''
async def async_call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 2. 运行对话
print_section("运行对话")
query = "Hi! I'm Bob."

'''
output["messages"][-1] 中的 -1 表示访问列表的最后一个元素。这是一种常用的Python列表切片语法：
-1 表示从末尾开始的第一个元素
-2 表示从末尾开始的第二个元素
'''
input_messages = [HumanMessage(query)]
config = {"configurable": {"thread_id": "abc123"}}
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
config = {"configurable": {"thread_id": "abc123"}} # 改变thread_id就创建一个新的对话, 不会和之前的对话连接
output = app.invoke({"messages": input_messages}, config)
output["messages"][-2].pretty_print()
output["messages"][-1].pretty_print()
