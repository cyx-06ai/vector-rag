from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from config import DEEPSEEK_KEY, SILICONFLOW_KEY
import random
from datetime import datetime

app = FastAPI()

# ===== 备货：启动时干一次（建知识库 + 造 Agent） =====
embeddings = OpenAIEmbeddings(
    api_key=SILICONFLOW_KEY, base_url="https://api.siliconflow.cn/v1",
    model="BAAI/bge-m3", check_embedding_ctx_length=False)
f = open("knowledge.txt", "r", encoding="utf-8")
lines = [line for line in f.read().split("\n") if line != ""]
f.close()
vectorstore = InMemoryVectorStore.from_texts(lines, embeddings)

def search_knowledge(query: str) -> str:
    """查询本地知识库，获取编程、AI、公司介绍等知识性资料"""
    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "知识库中没有相关资料"
    return "\n".join(d.page_content for d in docs)

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

agent = create_react_agent(
    model,
    [search_knowledge, roll_dice, get_time, get_weather],
    prompt="你是一个助手。需要实时信息或知识性资料时调用工具；回答知识问题时只能依据工具返回的资料，资料里没有就明说，禁止编造。")
histories = {}     # 本子柜：{访客编号：那个人的对话历史}

# ===== 接单：每次访问干一次 =====
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/ask") 
def ask(q: str, visitor: str = "default"):
    # 按编号拿本子；新访客自动发一本新的
    history = histories.setdefault(visitor, [])
    # 金鱼修复：备忘录历史 + 新问题，一起交给 Agent
    result = agent.invoke({"messages": history + [("user", q)]})
    reply = result["messages"][-1].content
    # 双写：把这轮全部记录 (含点单过程) 抄回备忘录
    history.extend(result["messages"])
    return {"reply": reply}