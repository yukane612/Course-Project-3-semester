from ollama import chat

prompt = "1+1=？"

res = chat(
    model="qwen:7b",
    messages=[
        {"role":"system","content":"你是一个专业的数学老师"},
        {"role":"user","content":prompt}
    ],
    stream = True
)

for chunk in res:
    print(chunk["message"]["content"],end = "")