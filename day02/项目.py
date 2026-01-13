import pandas as pd
from zhipuai import ZhipuAI
import pymysql
import gradio as gr

client = ZhipuAI()


def query_mysql(sql):
    # 连接MySQL语句
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="123",
        port=3306,
        database="newsdb",
        charset="utf8mb4"
    )
    # 连接MySQL
    cursor = conn.cursor()
    # 执行SQL语句
    cursor.execute(sql)
    # 获取执行的结果
    rows = cursor.fetchall()
    # 关闭连接
    conn.close()
    return rows

#获取数据，连入ai
def run_classify():
    #获取所有的news数据
    new_list = query_mysql("select id,title from news")
    #获取分类数据
    category_list = query_mysql("select name from category")
    #将category里面数据一个个获取到，使用列表推导式
    category = [l[0] for l in category_list]

    #定义一个存储分类完毕的变量
    results = []

    for new_id,title in new_list:
        prompt = f"""
        将下面新闻分类到以下类别之一，只返回类别名称[{category}]
        
        示例:
        清华大学公布2023年自主招生简章 => 教育
        
        新闻:
        {title}
        
        限制:
        严格按示例格式进行输出
        
        """

        #将这个prompt与ai交互
        res = client.chat.completions.create(
            model="glm-4-plus",
            messages=[
                {"role":"user","content":prompt}
            ]
        )

        #将这个数据交给results
        results.append([new_id,title,res.choices[0].message.content])

        #在pandas里渲染
        df = pd.DataFrame(results,columns=["编号","新闻","分类"])
        # 将数据流式输出显示,在gradio当中,如果我们想要实现流式输出,那么使用yield
        yield df

with gr.Blocks() as demo:
    gr.Markdown("#<center> 🤖 智能新闻分类系统</center>")

    btn = gr.Button("🧑🏼‍🚀 开始分类")

    show = gr.DataFrame(headers=["编号","新闻","分类"])

    btn.click(fn=run_classify,outputs=show)
demo.launch()


