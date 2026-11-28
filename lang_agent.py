from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import DEEPSEEK_KEY
import random
from datetime import datetime

# ===== 三个工具：只写函数本体，说明书自动生成 =====
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

# ===== 一行造出整个 Agent：菜单、分单员、循环全打包 =====
agent = create_react_agent(model, [roll_dice, get_time, get_weather])

while True:
    question = input("你：")
    if question == "退出":
        break
    result = agent.invoke({"messages": [("user", question)]})
    print("AI：", result["messages"][-1].content)