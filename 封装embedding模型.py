from zhipuai import ZhipuAI
client = ZhipuAI()

#input后面跟的是数组，传进来的text也是一个数组

def get_embedding(text):
    result = client.embeddings.create(
        input = text,
        model= "embedding-3"
    )
    return [x.embedding for x in result.data]

query = ["深度学习是机器学习里面的一个分支","你好啊"]
res = get_embedding(query)
print(res[0])
print(res[1])
print(len(res[0]))
print(len(res[1]))
