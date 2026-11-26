from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from config import DEEPSEEK_KEY, SILICONFLOW_KEY

embeddings = OpenAIEmbeddings(
    api_key=SILICONFLOW_KEY,
    base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3",
    check_embedding_ctx_length=False
)

f = open("knowledge.txt", "r", encoding="utf-8")
lines = [line for line in f.read().split("\n") if line !=""]
f.close()

print("正在把知识库变向量, 稍等......")
vectorstore = InMemoryVectorStore.from_texts(lines, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

template = ChatPromptTemplate.from_messages([
    ("system", "你只能根据下面的资料回答问题,  资料里没有就回答: 资料里没有。禁止编造。\n资料:\n{info}"),
    ("user", "请回答: {question}")
])

chain = template | model

q  = "DeepSeek是什么"
docs = retriever.invoke(q)
info = ""
for d in docs:
    info = info + d.page_content + "\n"

print("======查到的资料======")
print(info)    

reply = chain.invoke({"info": info, "question": q})    
print(reply.content)

