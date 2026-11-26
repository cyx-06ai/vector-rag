from langchain_openai import ChatOpenAI
from config import DEEPSEEK_KEY

model = ChatOpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

reply = model.invoke("用一句话介绍你自己")
print(reply.content)