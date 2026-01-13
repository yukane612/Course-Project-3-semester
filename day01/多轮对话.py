from zhipuai import ZhipuAI

client = ZhipuAI()

history = []

def func(prompt):

    history.append({"role":"user","content":prompt})

    res = client.chat.completions.create(
        model="glm-4",
        messages= history,
        stream = True
    )
    full_message = ""

    for chunk in res:
        content = chunk.choices[0].delta.content
        print(content,end="")
        full_message += content

while True:
    user_input = input("\n请输入问题")
    if user_input in ["退出"]:
        print("对话终止...")
        break
    func(user_input)