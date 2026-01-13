from numpy import  dot
import numpy as np
from zhipuai import ZhipuAI
from numpy.linalg import norm

client = ZhipuAI()

#计算余弦距离，越大越相似
def cos_sim(a,b):# 这个不需要背,了解就可以了
    return dot(a,b) / (norm(a) * norm(b))

#计算欧式距离,越小越相似
def l2(a,b): # 这个不需要背,了解就可以了
    return norm(np.asarray(a) - np.asarray(b))


#封装向量模型
def get_embedding(text):
    res = client.embeddings.create(
        model = "embedding-3",
        input = text
    )
    return [x.embedding for x in res.data]

#定义文本

query = "类司99"
documents = [
    "类司好甜",
    "谢谢你给我改变的契机",
    "在遥远宇宙中与你坠入爱河"
]
#先把两个转换成向量
q_results = get_embedding([query])[0]
d_results = get_embedding(documents)

#先跟自己作对比
#1.用余弦距离计算
print(cos_sim(q_results,q_results))
print("----------------------------")
for i in d_results:
    print(cos_sim(q_results,i))

#2.用欧式距离计算
print("===========================")
print(l2(q_results,q_results))
print("----------------------------")
for r in d_results:
    print(l2(q_results,r))