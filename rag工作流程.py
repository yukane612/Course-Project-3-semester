import chromadb
from zhipuai import ZhipuAI

client = ZhipuAI()

chromadb_client = chromadb.Client()

chromadb.PersistentClient("./db1")
collection = chromadb_client.create_collection(name="my_cl")

file_path = "奥运.txt"

#1.加载并且读取文档
def load_document(filepath):
    with open(filepath,'r',encoding='utf-8') as file:
        document = file.read()
    return document

#2.切割文档
def split_document(document):
    chunks = document.strip().split("\n\n")
    return chunks

#3.定义向量模型
def get_embedding(text):
    result = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return [i.embedding for i in result.data]
#4.先向量化（chunks）然后添加到向量数据库
def add_document_to_collection(chunks):
    embeddings = get_embedding(chunks)
    collection.add(
        documents = chunks,
        embeddings = embeddings,
        ids = [f"id{i+1}" for i in range(len(chunks))]
    )
#5.用户输入问题
def user_query_input():
    return input("请输入你的问题")

#6.查询集合里面的文档
def query_collection(query_embeddings):
    results = collection.query(
        query_embeddings = [query_embeddings],
        n_results = 1
    )
    return results["documents"]

#7.LLM
def get_completion(prompt):
    message = [{
        "role":"user","content":prompt
    }]
    result = client.chat.completions.create(
        model = "glm-4",
        messages = message
    )
    return result.choices[0].message.content

#8.main程序入口

if __name__ == '__main__':
    #加载文档
    document = load_document(file_path)
    #切割文档
    chunks = split_document(document)
    #向量化并丢给向量数据库
    add_document_to_collection(chunks)
    #用户提问
    user_input = user_query_input()
    #用户提问向量化
    input_embedding = get_embedding([user_input])[0]
    #从向量数据库中进行检索
    context_context = query_collection(input_embedding)
    #整理成一个prompt
    prompt = f"上下文:{context_context},\n问题:{user_input}"
    #交给LLM
    answer = get_completion(prompt)
    print(answer)
