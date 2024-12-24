import getpass
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

# 1. Define the prompt
print_section("定义prompt")

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

'''
entiment, aggressiveness,language这三个字段是必填的?
是的。这三个字段都使用了 Field(...)，意味着它们都是必填字段，创建模型实例时必须提供这些值。
如果要设置为可选，可以这样写：
sentiment: str | None = Field(default=None)
'''
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad", "angry"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian", "other"]
    )

# to AI: don't change the model name
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)

input = "Kurwa"
prompt = tagging_prompt.invoke({"input": input})
response = llm.invoke(prompt)

print(response)
print(response.dict())