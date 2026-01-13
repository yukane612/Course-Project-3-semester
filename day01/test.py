# 1. 导包,将我们刚刚下载好的智谱AI的包导入进来
from zhipuai import ZhipuAI

# 创建一个客户端
client = ZhipuAI()
# 我们要和AI进行交互
res = client.chat.completions.create(
    model="glm-4",# 智谱AI有很多的模型,我们使用glm-4,稳定有便宜
    messages=[
        # 给智谱AI定义一个系统角色,告诉智谱AI你是谁
        {"role":"system","content":"prompt"},
        # user是用户,我们就是用户,我们问AI的问题
        {"role":"user","content":"344+45=?"}
    ]
)
# 获取AI回复我们的结果,固定写法
print(res.choices[0].message.content)
