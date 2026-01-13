from zhipuai import ZhipuAI

instruction="""

你的任务就是识别用户的信息,提取时间(time),地点(location),人物(person),动作(action)

"""

example = """
示例：
下个周末，我将和我的同事王五一起去海洋公园玩耍
{“time”:"下个周末","location":"海洋公园","person":"我和王五","action":"去玩耍" }

"""


input_txt= """
输入：
小琳和小璐这个星期打算选个具体时间去菲尼克斯乐园玩
沈大雷和田马斯在周六完成了记牢大学习

"""

output = """
限制：
以JSON格式输出
"""
#组装成一prompt => f 表达式 =>

prompt = f"""
{instruction}
{example}
{input_txt}
{output}

"""

client = ZhipuAI()
res = client.chat.completions.create(
    model= "glm-4",
    messages =[
        {"role":"user","content":prompt}
    ]
)

print(res.choices[0].message.content)