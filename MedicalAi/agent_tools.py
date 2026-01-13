from zhipuai import ZhipuAI
import json
from MedicalAi.rag_core import retrieve_similar_docs, init_vector_db
from config import LLM_MODEL, SENSITIVE_WORDS, CAMPUS_KNOWLEDGE

conversation_history = []
# 初始化智谱客户端
client = ZhipuAI()

# ====================== 工具定义======================
tools = [
    # 工具1：症状追问工具（信息不足时调用）
    {
        "type": "function",
        "function": {
            "name": "symptom_inquiry",
            "description": "当用户仅描述症状（如头痛、发烧），信息不足无法提供科普时，调用此工具追问关键细节",
            "parameters": {
                "type": "object",
                "properties": {
                    "inquiry_content": {
                        "type": "string",
                        "description": "追问的具体问题，如「头痛是否伴随恶心？发烧体温是多少？」"
                    }
                },
                "required": ["inquiry_content"]
            }
        }
    },
    # 工具2：专科推荐工具（命中敏感词时调用）
    {
        "type": "function",
        "function": {
            "name": "department_recommend",
            "description": "当用户问题包含敏感词（如确诊、癌症），需推荐对应专科就诊，禁止提供科普信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "推荐的专科名称，如「肿瘤科」「心血管内科」"
                    },
                    "suggestion": {
                        "type": "string",
                        "description": "就诊建议，如「请尽快前往三甲医院肿瘤科就诊，携带既往病历」"
                    }
                },
                "required": ["department", "suggestion"]
            }
        }
    },
    # 工具3：校园咨询工具（用户问校园相关问题时调用）
    {
        "type": "function",
        "function": {
            "name": "campus_consult",
            "description": "当用户问题是校园管理相关（如图书馆开放时间、宿舍申请），调用此工具返回答案",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "校园问题的准确答案，直接引用校园知识库"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


# ====================== 工具执行函数======================
def execute_tool(tool_name, tool_args):
    """执行工具调用，返回工具结果"""
    if tool_name == "symptom_inquiry":
        return tool_args["inquiry_content"]  # 直接返回追问内容
    elif tool_name == "department_recommend":
        return f"建议专科就诊：{tool_args['department']}。{tool_args['suggestion']}"
    elif tool_name == "campus_consult":
        return tool_args["answer"]
    else:
        return "暂无相关工具可执行"


# ====================== 敏感词检测（合规强化）======================
def check_sensitive_words(text):
    """检测文本中是否包含敏感词，返回检测结果和敏感词"""
    for word in SENSITIVE_WORDS:
        if word in text:
            return True, word
    return False, None


# ====================== Agent核心逻辑======================
def agent_answer(user_query, role="患者"):
    """Agent核心函数：整合RAG检索+工具调用"""
    global conversation_history
    # 步骤1：敏感词检测
    has_sensitive, sensitive_word = check_sensitive_words(user_query)
    if has_sensitive:
        return f"涉及敏感内容「{sensitive_word}」，建议专科就诊：相关疾病需专业医生诊断，请勿自行判断，尽快前往三甲医院对应科室就诊。"

    # 步骤2：校园咨询检测
    # 定义校园关键词的同义扩展字典（覆盖所有校园场景）
    campus_synonyms = {
        "如何申请宿舍": ["申请宿舍", "宿舍申请", "怎么申请宿舍", "宿舍怎么申请"],
        "图书馆开放时间": ["图书馆什么时候开", "图书馆开放时间", "图书馆开门时间", "图书馆几点开"],
        "医保报销流程": ["医保怎么报销", "医保报销流程", "报销医保", "医保报销怎么弄"],
        "成绩单打印": ["打印成绩单", "成绩单怎么打印", "成绩单打印流程"]
    }

    # 遍历所有校园场景，模糊匹配同义关键词
    campus_answer = ""
    for standard_key, synonyms in campus_synonyms.items():
        # 检查用户问题是否包含任意一个同义关键词
        if any(synonym in user_query for synonym in synonyms):
            campus_answer = f"校园咨询回复：{CAMPUS_KNOWLEDGE[standard_key]}"
            break

    # 匹配到校园场景，直接返回（不追加医疗合规提示）
    if campus_answer:
        return campus_answer

    # 步骤3：RAG检索医疗知识库（不变）
    collection = init_vector_db()
    retrieved_docs = retrieve_similar_docs(collection, user_query)

    # 步骤4：构造Agent Prompt（含角色适配、工具定义、合规规则）
    role_prompt = "用通俗的语言解释，避免专业术语，语气亲切" if role == "患者" else "用规范医学术语回答，简洁专业，突出核心信息"

    # 拼接对话历史：将之前的交互记录+当前问题+参考资料整合到Prompt
    # 先清空历史中过长 的记录（避免Token超限，保留最近5轮，参考文档“滑动窗口”思路）
    if len(conversation_history) > 10:  # 每轮2条（用户+助手），保留5轮
        conversation_history = conversation_history[-10:]

    prompt = f"""
    你是合规的医疗信息助手，{role_prompt}，可调用以下工具：
    {json.dumps(tools, ensure_ascii=False, indent=2)}
    
    以下是之前的对话历史，请结合历史理解当前问题：
    {json.dumps(conversation_history, ensure_ascii=False)}

    严格遵守以下规则：
    1. 禁止做出疾病诊断、治疗建议，仅基于参考资料提供医疗科普信息；
    2. 必须引用参考资料中的内容，不编造信息；
    3. 结合对话历史理解用户意图（如用户先问“儿童发烧怎么办”，再问“用什么药”，指“儿童发烧用什么药”）；
    4. 回答末尾必须强制追加：「以上信息仅供参考，如有不适请及时线下就医」；
    5. 工具调用格式必须为JSON：{{"tool": "工具名", "args": {{"参数名": "参数值"}}}}，禁止其他格式。
    
    参考资料：
    {retrieved_docs}
    
    当前用户问题：{user_query}
    """

    # 步骤5：调用GLM模型，获取响应（可能是回答或工具调用指令）
    # 先将对话历史+当前Prompt整理为messages格式
    messages = conversation_history.copy()  # 复制历史消息
    messages.append({"role": "user", "content": prompt})  # 追加当前Prompt
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        temperature=0.7
    )

    # 步骤6：解析响应（判断是直接回答还是工具调用）
    response_msg = response.choices[0].message
    if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
        # 有工具调用指令，解析并执行
        tool_call = response_msg.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # 执行工具，获取结果
        tool_result = execute_tool(tool_name, tool_args)

        # 步骤6：将工具结果追加到上下文，生成最终回答
        messages.append(response_msg.model_dump())  # 智谱需加model_dump()（文档重点提示）
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_name,
            "content": tool_result
        })

        # 二次调用模型，生成最终回答
        final_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7
        )
        answer = final_response.choices[0].message.content
    else:
        # 无工具调用，直接使用模型回答
        answer = response_msg.content

    # 步骤7：合规校验（确保末尾有就医提示）
    if answer.strip() and "校园咨询回复" not in answer:
        # 仅医疗回答且内容非空时追加
        if "以上信息仅供参考，如有不适请及时线下就医" not in answer:
            answer += "\n\n以上信息仅供参考，如有不适请及时线下就医"

    # 步骤8：更新对话历史（存入当前用户问题+助手回答）
    conversation_history.append({"role": "user", "content": user_query})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer


# 测试Agent功能
if __name__ == "__main__":
    print("测试1（症状信息不足）：", agent_answer("头痛", "患者"))
    print("测试2（医生角色）：", agent_answer("儿童发烧39℃怎么办？", "医生"))
    print("测试3（校园咨询）：", agent_answer("图书馆开放时间？", "患者"))
    print("测试4（敏感词）：", agent_answer("我被确诊癌症了", "患者"))