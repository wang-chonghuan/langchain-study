import getpass
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

# 添加分隔函数
def print_section(section_name):
    print("\n" + "="*50)
    print(f"开始执行: {section_name}")
    print("="*50 + "\n")

# 0. Load the documents
print_section("加载文档")

# 加载环境变量
load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

file_path = "./data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))

print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

# 1. Split the documents into chunks
print_section("分割文档")
'''
在分割文本时，add_start_index=True记录每个分割块在其所属页面中的字符起始位置.这个位置信息会被存储在 metadata 的 start_index 字段中.记住,这不是该chunk在整个文档中的字符起始位置.
'''
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"Number of chunks: {len(all_splits)}")

print(f"{all_splits[0].page_content[:200]}\n")
print(all_splits[513].metadata)
print(all_splits[514].metadata)
print(all_splits[515].metadata)

# 2. Create embeddings for the chunks
print_section("创建文档嵌入")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
print(vector_2[:10])

# 3. Create a vector store
'''
to AI: don't delete this comment
https://python.langchain.com/docs/integrations/vectorstores/faiss/
'''
print_section("创建向量存储")
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

ids = vector_store.add_documents(documents=all_splits)

print(f"Number of vectors in the vector store: {vector_store.index.ntotal}")

# 4. Query the vector store

# 4.1 相似度搜索
results = vector_store.similarity_search("How many distribution centers does Nike have in the US?")
print(results[0])

# 4.2 相似度搜索并返回分数
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

# 4.3 向量搜索
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results1 = vector_store.similarity_search_by_vector(embedding)
print(results1[0])

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

results2 = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print('-'*50)
print(results2)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

results3 = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print('-'*50)
print(results3)