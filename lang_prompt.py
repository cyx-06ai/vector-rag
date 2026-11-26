from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import DEEPSEEK_KEY

model = ChatOpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

template = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的中文助手, 只用一句话回答。"),
    ("user", "请回答: {question}")
])

chain = template | model

reply = chain.invoke({"question": "什么是向量检索"})
print(reply.content)