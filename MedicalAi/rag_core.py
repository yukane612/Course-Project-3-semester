import chromadb
from config import VECTOR_DB_PATH, TOP_K
from utils import get_embedding, call_glm
from medical_knowledge import medical_chunks


def init_vector_db():
  """
  向量库初始化：持久化存储，首次运行创建，后续直接加载
  :return: Chroma的collection对象（向量库集合）
  """
  # 1. 初始化持久化客户端（数据保存在VECTOR_DB_PATH，重启不丢失）
  chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

  # 2. 创建/加载集合
  collection = chroma_client.get_or_create_collection(
    name = "medical_collection",
    metadata = {"description": "医疗问答知识库"}
  )

  # 3. 仅首次运行时导入数据（避免重复导入）
  if collection.count() == 0:
    print("首次运行，导入医疗知识库到向量库...")
    # 生成向量
    embeddings = get_embedding(medical_chunks)
    if not embeddings:
      raise Exception("向量生成失败，无法初始化向量库")

    # 生成唯一ID（如doc_0, doc_1...）
    ids = [f"doc_{i}" for i in range(len(medical_chunks))]

    # 导入向量库
    collection.add(
      documents = medical_chunks,
      embeddings = embeddings,
      ids = ids
    )
    print(f"导入完成！共{len(medical_chunks)}条医疗片段，{collection.count()}条向量")
  else:
    print(f"向量库已存在，共{collection.count()}条向量")

  return collection


def retrieve_similar_docs(collection, query_text):
    """
    检索相似文档：根据用户问题找Top-K相关片段
    :param collection: Chroma集合对象
    :param query_text: 用户问题（如“儿童发烧39℃怎么办？”）
    :return: 格式化的检索结果（参考资料）
    """
    # 1. 问题向量化
    query_embedding = get_embedding([query_text])
    if not query_embedding:
        return "检索失败：问题向量生成错误"

    # 2. 向量库检索
    results = collection.query(
        query_embeddings = query_embedding,
        n_results = TOP_K,  # 返回Top-2相似文档
        include = ["documents", "distances"]  # 包含文档内容和相似度距离
    )

    # 3. 格式化检索结果
    similar_docs = results["documents"][0]  # 第0个查询的结果
    distances = results["distances"][0]  # 相似度距离（越小越相似）

    if not similar_docs:
        return "未检索到相关医疗信息"

    formatted_result = "【参考资料】\n"
    for i, (doc, dist) in enumerate(zip(similar_docs, distances)):
        # 距离<0.8视为相关（可调整阈值）
        relevance = "高" if dist < 0.8 else "中" if dist < 1.0 else "低"
        formatted_result += f"{i + 1}. 相似度：{relevance}（距离{dist:.3f}）\n{doc}\n\n"

    return formatted_result


def rag_answer(collection, query_text):
    """
    RAG流程：检索→构造Prompt→生成回答
    :param collection: Chroma集合对象
    :param query_text: 用户问题
    :return: 最终回答（含参考资料+合规提示）
    """
    # 1. 检索相似文档
    retrieved_docs = retrieve_similar_docs(collection, query_text)

    # 2. 构造Prompt
    prompt = f"""
    你是合规的医疗信息助手，严格遵守以下规则：
    1. 禁止做出疾病诊断、治疗建议，仅基于参考资料提供科普信息；
    2. 必须引用参考资料中的内容，不编造信息；
    3. 回答末尾强制追加：「以上信息仅供参考，如有不适请及时线下就医」；
    4. 若参考资料无相关信息，仅回复「暂无相关医疗科普信息，建议咨询专业医生」。

    参考资料：
    {retrieved_docs}

    用户问题：{query_text}
    请生成简洁、易懂的回答：
    """

    # 3. 调用LLM生成回答
    answer = call_glm(prompt)

    # 4. 确保合规提示
    if "以上信息仅供参考，如有不适请及时线下就医" not in answer:
        answer += "\n\n以上信息仅供参考，如有不适请及时线下就医"

    return answer