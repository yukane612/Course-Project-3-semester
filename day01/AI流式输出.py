from zhipuai import ZhipuAI

client = ZhipuAI()
prompt = "给我一些你的经验和建议"
res = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role":"system","content":"你是一个专业的钢琴家"},
        {"role":"user","content":prompt}
    ],
    stream = True
)
for chunk in res:
    content = chunk.choices[0].delta.content
    print(content,end="")