from openai import OpenAI
from config import DEEPSEEK_KEY
import random
import json
from datetime import datetime

client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

# ===== 三个真工具（本地函数，代码执行） =====
def roll_dice(sides):
    return str(random.randint(1, sides))

def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_weather(city):
    weathers = {"上海": "晴，22度", "北京": "多云，15度", "广州": "小雨，26度"}
    return weathers.get(city, "暂无该城市的天气数据")

# ===== 菜单：给模型看的说明书数组 =====
tools = [
    {"type": "function", "function": {
        "name": "roll_dice",
        "description": "掷一个骰子",
        "parameters": {"type": "object", "properties": {
            "sides": {"type": "integer", "description": "骰子面数，比如6"}
        }, "required": ["sides"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "获取当前的日期和时间",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询某个城市今天的天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，比如上海"}
        }, "required": ["city"]}}},
]

# ===== 分单员：模型点哪道菜，就派给哪个函数 =====
def run_tool(name, args):
    if name == "roll_dice":
        return roll_dice(args["sides"])
    if name == "get_time":
        return get_time()
    if name == "get_weather":
        return get_weather(args["city"])
    return "没有这个工具"

# ===== 主循环 =====
while True:
    question = input("你：")
    if question == "退出":
        break

    messages = [
        {"role": "system", "content": "你是一个助手，需要实时信息时调用工具，不要编造数据。"},
        {"role": "user", "content": question}
    ]

    reply = client.chat.completions.create(messages=messages, tools=tools, model="deepseek-chat")
    msg = reply.choices[0].message

    # 只要模型还在下单，就一直干到它满意为止
    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = run_tool(tc.function.name, args)
            print(f"   [干活：{tc.function.name}({args}) → {result}]")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        reply = client.chat.completions.create(messages=messages, tools=tools, model="deepseek-chat")
        msg = reply.choices[0].message

    print("AI: ", msg.content)