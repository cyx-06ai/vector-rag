from openai import OpenAI
from config import DEEPSEEK_KEY, SILICONFLOW_KEY

client = OpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com"
)
embed_client = OpenAI(
    api_key=SILICONFLOW_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

def get_vec(text):
    response = embed_client.embeddings.create(
        model="BAAI/bge-m3",
        input=text
    )
    return response.data[0].embedding

def similarity(v1, v2):
    total = 0
    for i in range(len(v1)):
        total = total + v1[i] * v2[i]
    return total

f = open("knowledge.txt", "r", encoding="utf-8")
text = f.read()
f.close()
lines = text.split("\n")

print("正在把知识库变向量,稍等......")
line_vecs = []
for line in lines:
    line_vecs.append(get_vec(line))

messages=[
        {"role": "system", "content":"你只能根据下面给的资料回答问题,资料里没有的就回答:资料里没有。禁止编造。"}]

while True:
    question = input("你: ")
    if question == "退出":
        break
    q_vec = get_vec(question)

    pairs = []
    for i in range(len(lines)):
        score = similarity(q_vec, line_vecs[i])
        pairs.append([score,lines[i]])

    pairs.sort(reverse=True)
    top3 = pairs[:3]

    info = ""
    for p in top3:
        if p[0] > 0.5:
            info = info + p[1] + "\n"

    messages.append({"role": "user", "content": "资料:\n" + info + "\n问题:" + question})

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages
    )
    reply = response.choices[0].message.content
    print("AI:", reply)
    messages.append({"role": "assistant", "content": reply})