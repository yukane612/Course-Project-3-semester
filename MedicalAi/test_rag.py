from MedicalAi.rag_core import init_vector_db, rag_answer
from utils import split_document

def main():
    # 1. 初始化向量库（首次运行导入数据，后续直接加载）
    try:
        collection = init_vector_db()
    except Exception as e:
        print(f"向量库初始化失败：{str(e)}")
        return

    # 2. 测试RAG流程
    test_queries = [
        "儿童发烧39℃怎么办？",
        "阿司匹林能和布洛芬一起吃吗？",
        "布洛芬的禁忌是什么？",
        "儿童腹泻有脓血便该怎么办？",
        "肺癌怎么治疗？"  # 测试无相关信息的情况
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 测试{i}：{query} ===")
        answer = rag_answer(collection, query)
        print("AI回答：")
        print(answer)


if __name__ == "__main__":
    main()