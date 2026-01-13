from zhipuai import ZhipuAI
client = ZhipuAI()
res = client.embeddings.create(
    model="embedding-2",
    input=[
        "雾草，产品好甜",
        "谢谢你给我改变的契机",
        "在遥远宇宙中与你坠入爱河"
    ]
)

#获取第一句话，把res里的数据循环取出来（res。data）
print([x.embedding for x in res.data][0])
print([x.embedding for x in res.data][1])
print([x.embedding for x in res.data][2])
#求长度
print(len([x.embedding for x in res.data][0]))
print(len([x.embedding for x in res.data][1]))
print(len([x.embedding for x in res.data][2]))





