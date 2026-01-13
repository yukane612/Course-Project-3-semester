from zhipuai import ZhipuAI
client = ZhipuAI()

user_query = "儿童发烧如何处理"
#零样本
prompt_zero = f"请回答这个问题:{user_query}"

#少样本
prompt_few = """
示例：

问题：

限制


"""

#思维链cot

