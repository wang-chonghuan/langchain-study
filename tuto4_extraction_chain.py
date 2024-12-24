import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 添加分隔函数
def print_section(section_name):
    print("\n" + "="*50)
    print(f"开始执行: {section_name}")
    print("="*50 + "\n")

# 0. Load the documents
print_section("加载环境变量")

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# 1. Define the schema
print_section("定义schema")

'''
doc-string 注释会成为 schema 的整体描述
字段的 description 会成为每个字段的具体说明
这些信息会被组合成一个结构化的提示发送给 LLM
LLM 根据这些说明从文本中提取相关信息
For best performance, document the schema well and make sure the model isn't force to return results if there's no information to be extracted in the text.
'''
class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )

class Data(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[Person]

# 2. Define the prompt template
print_section("定义prompt template")

# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)


# 3. Define the LLM
print_section("定义LLM")

llm = ChatOpenAI(model="gpt-4o-mini")

structured_llm = llm.with_structured_output(schema=Person)
text = "Alan Smith is 6 feet tall and has blond hair."
prompt = prompt_template.invoke({"text": text})
response = structured_llm.invoke(prompt)
print(response)

structured_llm1 = llm.with_structured_output(schema=Data)
text1 = "car is running"
prompt1 = prompt_template.invoke({"text": text1})
response1 = structured_llm1.invoke(prompt1)
print(response1)




# 4. 定义工具调用
print_section("定义工具调用")

from langchain_core.utils.function_calling import tool_example_to_messages

few_shot_examples = [
    (
        "The ocean is vast and blue. It's more than 20,000 feet deep.",
        Data(people=[]),
    ),
    (
        "Fiona traveled far from France to Spain.",
        Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
    ),
]

few_shot_examples_messages = []

for txt, tool_call in few_shot_examples:
    if tool_call.people:
        # This final message is optional for some providers
        ai_response = "Detected people."
    else:
        ai_response = "Detected no people."
    few_shot_examples_messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

for message in few_shot_examples_messages:
    message.pretty_print()

message_no_extraction = {
    "role": "user",
    "content": "The solar system is large, but earth has only 1 moon.",
}

'''
将message_no_extraction放入[]数组中的原因是：
保持消息格式一致性：LangChain期望接收一个消息列表作为输入。即使只有一条消息，也需要将其作为列表传入。
'''
structured_llm2 = llm.with_structured_output(schema=Data)
response2 = structured_llm2.invoke([message_no_extraction])
print(response2)

response3 = structured_llm2.invoke(few_shot_examples_messages + [message_no_extraction])
print(response3)