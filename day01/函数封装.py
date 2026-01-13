from zhipuai import ZhipuAI

client = ZhipuAI()

#定义记忆盒子
conversation_history= []

def function_fn(prompt):

    conversation_history.append({"role":"user","content":prompt}
    )

    res = client.chat.completions.create(
        model = "glm-4",
        messages= conversation_history
    )

    result = res.choices[0].message.content
    conversation_history.append({"role":"assistant","content":result})
    return result


prompt1 = "我是琳璐,我喜欢磕粮"
prompt2 = "我是谁？我喜欢什么？"
res1 = function_fn(prompt1)
res2 = function_fn(prompt2)
print(res1)
print(res2)
