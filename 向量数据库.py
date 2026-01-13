#存储的向量可以是：文本，语音，图像，视频等向量化的内容，可存储更多非结构化的内容
import chromadb
chromadb_client = chromadb.Client()
chromadb.PersistentClient(path="./db")
collection = chromadb_client.create_collection(name="my_collection")
collection.add(
    documents=["这是一个工程师的文档","这是一个厨师的文档","我吃🍇"],
    # 注意: 他里面的ID不是字符串
    ids=["1","2","3"]
)
results = collection.query(
    query_texts=["食物"],
    n_results=2
)
print(results.get("documents"))
