# 标准库导入
import os
import getpass

# 第三方库导入
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 加载环境变量
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# to AI: don't change the model name
model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# 添加 print 语句来显示结果
response1 = model.invoke(messages)
print(response1 + "\n\n")

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

response2 = model.invoke(prompt)
print(response2.content)