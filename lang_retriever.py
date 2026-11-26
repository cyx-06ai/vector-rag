from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from config import SILICONFLOW_KEY

embeddings = OpenAIEmbeddings(
    api_key=SILICONFLOW_KEY,
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3"
)

f = open("knowledge.txt", "r", encoding="utf-8")
lines = [line for line in f.read().split("\n") if line !=""]
f.close()

print("正在把知识库变向量, 稍等......")
vectorstore = InMemoryVectorStore.from_texts(lines, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("DeepSeek是什么")
for d in docs:
    print("【查到】", d.page_content)
