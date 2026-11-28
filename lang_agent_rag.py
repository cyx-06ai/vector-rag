from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from config import DEEPSEEK_KEY, SILICONFLOW_KEY
import random
from datetime import datetime

# ===== 备货：启动时建知识库（跟 lang_rag.py 一样） =====
embeddings = OpenAIEmbeddings(
    api_key=SILICONFLOW_KEY, base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3", check_embedding_ctx_length=False)
f = open("knowledge.txt", "r", encoding="utf-8")
lines = [line for line in f.read().split("\n") if line != ""]
f.close()
vectorstore = InMemoryVectorStore.from_texts(lines, embeddings)

# ===== 工具1：查知识库（新菜！） =====
def search_knowledge(query: str) -> str:
    """查询本地知识库，获取编程、AI、公司介绍等知识性资料"""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "知识库中没有相关资料"
    return "\n".join(d.page_content for d in docs)

# ===== 工具2/3/4：昨天的三道老菜 =====
def roll_dice(sides: int) -> str:
    """掷一个骰子"""
    return str(random.randint(1, sides))

def get_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city: str) -> str:
    """查询某个城市今天的天气"""
    weathers = {"上海": "晴，22度", "北京": "多云，15度", "广州": "小雨，26度"}
    return weathers.get(city, "暂无该城市的天气数据")

model = ChatOpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com", model="deepseek-chat")

# ===== 四道菜上桌，第三个参数是给 Agent 立规矩 =====
agent = create_react_agent(
    model,
    [search_knowledge, roll_dice, get_time, get_weather],
    prompt="你是一个助手。需要实时信息或知识性资料时调用工具；回答知识问题时只能依据工具返回的资料，资料里没有就明说，禁止编造。")

while True:
    question = input("你：")
    if question == "退出":
        break
    result = agent.invoke({"messages": [("user", question)]})
    print("AI：", result["messages"][-1].content)