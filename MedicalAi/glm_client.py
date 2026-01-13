import os
from zhipuai import ZhipuAI

client = ZhipuAI()

def call_glm(prompt):
    try:
        response = client.chat.completions.create(
            model = "glm-4",
            messages = [
                {"role":"user","content":prompt}
            ],
            temperature=0.7,  # 平衡医疗信息的稳定性与自然度
            max_tokens=1024  # 避免模型生成超长冗余内容，同时控制API调用成本（按token计费）
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"调用失败：{str(e)}"

if __name__ == '__main__':
    test_prompt = """
    你是合规的医疗信息助手，严格遵守以下规则：
    1. 禁止做出疾病诊断、治疗建议，仅提供科普信息；
    2. 回答末尾必须强制追加：「以上信息仅供参考，如有不适请及时就医」；
    3. 回答简洁，分点说明即可，无需额外冗余内容。

    请回答：感冒吃什么药？
    """
    print(call_glm(test_prompt))