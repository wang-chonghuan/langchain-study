from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 添加分隔函数
def print_section(section_name):
    print("\n" + "=" * 50)
    print(f"开始执行: {section_name}")
    print("=" * 50 + "\n")

# 0. 加载环境变量
print_section("加载环境变量")

load_dotenv()

# 1. 创建工具
print_section("创建工具")

search = TavilySearchResults(max_results=2)
#search_results = search.invoke("what is the weather in SF")
#print(search_results)

# If we want, we can create other tools.
# Once we have all the tools we want, we can put them in a list that we will reference later.
tools = [search]

# 2. 测试工具调用
print_section("测试工具调用")

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke([HumanMessage(content="Hi!")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")

# 3. 创建反应式代理
print_section("创建反应式代理")
agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
print(response["messages"])

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")

# 4. 记忆功能
print_section("记忆功能")

config = {"configurable": {"thread_id": "abc123"}}

memory = MemorySaver()

agent_executor2 = create_react_agent(model, tools, checkpointer=memory)

for chunk in agent_executor2.stream(
    {"messages": [HumanMessage(content="hi im bob!")]}, config
):
    print(chunk)
    print("----")

for chunk in agent_executor2.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")

async def process_weather_query(agent_executor, query):
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content=query)]}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

async def main():
    print_section("异步事件流测试")
    query = "whats the weather in sf?"
    await process_weather_query(agent_executor, query)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())