f = open("knowledge.txt", "r", encoding="utf-8")
text = f.read()
f.close()

lines = text.split("\n")

q = input("你要找什么: ")

for line in lines:
    if q in line:
        print(line)