#封装嵌入、LLM 调用、文本分块
from zhipuai import ZhipuAI
from config import EMBEDDING_MODEL,LLM_MODEL

client = ZhipuAI()

def get_embedding(text_list):
    """
        智谱 embedding-3 模型调用：将文本列表转为向量
        :param text_list: 待向量化的文本列表（如["布洛芬适应症...", "儿童发烧处理..."]）
        :return: 向量列表（每个文本对应一个向量）
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text_list  # 必须传列表，单个文本需用[text]
        )
        # 提取向量（文档教学的返回格式解析）
        return [x.embedding for x in response.data]
    except Exception as e:
        print(f"嵌入模型调用失败：{str(e)}")
        return []


def call_glm(prompt):
    """
        调用智谱GLM生成回答
        :param prompt: 包含检索结果的Prompt
        :return: 模型生成的回答
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 文档未提，保留平衡值
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM调用失败：{str(e)}")
        return "抱歉，生成回答时出错，请重试。"

def split_document(text, chunk_size=200, chunk_overlap=20):
    """
    文档教学的文本分块：按长度切割，保留上下文重叠（适配医疗长文本）
    :param text: 原始长文本（如完整的药品说明书）
    :param chunk_size: 每个片段的最大长度
    :param chunk_overlap: 片段间重叠长度（避免语义断裂）
    :return: 分块后的文本列表
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        # 最后一个片段直接取到末尾
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
        # 避免在句子中间切割（找最近的句号/分号）
        end = text.rfind("。", start, end) or text.rfind("；", start, end) or end
        chunks.append(text[start:end+1].strip())
        # 下一段从重叠位置开始
        start = end + 1 - chunk_overlap
    return chunks