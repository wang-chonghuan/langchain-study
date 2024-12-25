import getpass
import os
from typing import Sequence
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    AIMessage, 
    HumanMessage, 
    BaseMessage,
    SystemMessage
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import trim_messages


# 添加分隔函数
def print_section(section_name):
    print("\n" + "=" * 50)
    print(f"开始执行: {section_name}")
    print("=" * 50 + "\n")


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

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 1. 定义一个状态图
print_section("定义一个状态图")

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define a new graph
# 对于无模板的聊天, 用workflow = StateGraph(state_schema=MessagesState)
workflow = StateGraph(state_schema=State)


# Define the function that calls the model
def call_model(state: State):
    # 此处传入state就行, 不需要state["messages"], 因为state里有多个属性可能prompt模板都需要
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": response}


# Async function for node:
"""
由于代码中使用了 await 关键字进行异步操作，app.ainvoke需要在异步函数（async函数）内部运行，并且要通过 asyncio 来执行，否则会报错。简单来说就是需要把代码包装在 async def main() 函数中，然后用 asyncio.run(main()) 来运行。
async def async_call_model(state: State):
    response = await model.ainvoke(state)
    return {"messages": response}
"""

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 2. 运行对话
print_section("运行对话")
query = "what is my name?"
language = "English"

"""
output["messages"][-1] 中的 -1 表示访问列表的最后一个元素。这是一种常用的Python列表切片语法：
-1 表示从末尾开始的第一个元素
-2 表示从末尾开始的第二个元素
"""
input_messages = messages + [HumanMessage(query)]
config = {"configurable": {"thread_id": "abc123"}}
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()  # output contains all messages in state

# 3. stream返回
print_section("stream返回")

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="")
