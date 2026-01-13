import gradio as gr
import pandas as pd
import numpy as np
from agent_tools import agent_answer, conversation_history
from config import CAMPUS_KNOWLEDGE
from rag_core import init_vector_db, retrieve_similar_docs
from medical_knowledge import medical_raw_data
import os
import tempfile
import time


# 清空对话历史的函数
def clear_conversation_history():
    global conversation_history
    conversation_history = []


# ===================== 核心功能函数 =====================
def load_medical_knowledge_to_df():
    """将医疗知识库转为DataFrame并渲染"""
    if isinstance(medical_raw_data, list):
        formatted_data = []
        for item in medical_raw_data:
            if "药物名称" in item:
                key = item["药物名称"]
            elif "场景" in item:
                key = item["场景"]
            else:
                key = "其他医疗信息"

            content = "; ".join([f"{k}：{v}" for k, v in item.items()])
            formatted_data.append({"场景/药物": key, "内容": content})

        df_medical = pd.DataFrame(formatted_data)
    else:
        df_medical = pd.DataFrame({"场景/药物": ["暂无数据"], "内容": ["暂无医疗知识库数据"]})

    return df_medical.fillna("无")


def load_campus_knowledge_to_df():
    """将校园知识库转为DataFrame并渲染"""
    df_campus = pd.DataFrame(
        list(CAMPUS_KNOWLEDGE.items()),
        columns=["问题", "回答"]
    )
    return df_campus


def export_chat_history_to_txt(chat_history):
    """导出对话记录为TXT"""
    if not chat_history:
        # 无对话记录时返回提示文件
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix="empty_chat_",
            delete=False,
            encoding="utf-8"
        )
        temp_file.write("暂无对话记录可导出！")
        temp_file.close()
        return temp_file.name

    # 整理对话记录
    export_content = "医疗+校园双场景问答系统 - 对话记录\n"
    export_content += "=" * 60 + "\n"
    user_msgs = []
    bot_msgs = []
    for msg in chat_history:
        if msg["role"] == "user":
            user_msgs.append(msg["content"])
        elif msg["role"] == "assistant":
            bot_msgs.append(msg["content"])

    # 生成对话内容
    for idx, (u, b) in enumerate(zip(user_msgs, bot_msgs), 1):
        export_content += f"\n【第{idx}轮】\n用户：{u}\n助手：{b}\n"

    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=f"chat_record_{time.strftime('%Y%m%d_%H%M%S')}_",
        delete=False,
        encoding="utf-8"
    )
    temp_file.write(export_content)
    temp_file.close()

    return temp_file.name


def clear_chat():
    """清空所有内容"""
    clear_conversation_history()
    return [], "", None, ""


def submit_msg(user_msg, role, chat_history):
    """提交消息核心逻辑"""
    if not user_msg.strip():
        return chat_history, "⚠️ 请输入有效问题！", ""

    try:
        collection = init_vector_db()
        retrieved_docs = retrieve_similar_docs(collection, user_msg)
        bot_msg = agent_answer(user_msg, role)
        new_chat = chat_history.copy()
        new_chat.append({"role": "user", "content": user_msg})
        new_chat.append({"role": "assistant", "content": bot_msg})
        ref_info = retrieved_docs if "校园咨询" not in bot_msg else "📚 校园知识库来源：宿舍申请/图书馆/医保报销"
        return new_chat, ref_info, ""
    except Exception as e:
        return chat_history, f"❌ 处理失败：{str(e)}", ""


# ===================== 自定义CSS样式 =====================
custom_css = """
.gradio-container {
    max-width: 1400px !important;
    font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
}
.title-section {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}
.card {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    background-color: white;
    border: 1px solid #f0f0f0;
}
.chat-container {
    min-height: 500px;
}
/* 控制Dataframe高度 */
.dataframe-container {
    max-height: 400px !important;
    overflow-y: auto !important;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    color: #666;
    font-size: 0.9rem;
    padding: 1rem 0;
    border-top: 1px solid #eee;
}
.gr-button-primary {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border: none !important;
}
.gr-button-secondary {
    background: #f8fafc !important;
    color: #334155 !important;
    border: 1px solid #e2e8f0 !important;
}
.dataframe {
    border-collapse: collapse !important;
    width: 100% !important;
}
.dataframe th {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    text-align: left !important;
    border-bottom: 2px solid #e2e8f0 !important;
}
.dataframe td {
    padding: 0.8rem !important;
    border-bottom: 1px solid #f0f0f0 !important;
}
"""

# ===================== Gradio界面搭建（适配6.0+） =====================
with gr.Blocks(title="医疗+校园双场景智能问答系统") as demo:
    # 1. 标题区域
    with gr.Column(elem_classes="title-section"):
        gr.Markdown("""
        ## 🧑‍⚕️ 医疗+校园双场景智能问答系统
        ### 🔍 基于RAG+Agent架构 | 合规医疗科普 | 智能校园咨询
        📌 支持TXT格式导出 | 知识库可视化查看
        """)

    # 2. 主内容区
    with gr.Tabs():
        # 2.1 对话标签页
        with gr.TabItem("💬 智能问答"):
            with gr.Row():
                # 左侧对话区
                with gr.Column(scale=3):
                    with gr.Column(elem_classes="card chat-container"):
                        chatbot = gr.Chatbot(
                            label="对话记录",
                            height=500,
                            show_label=False
                        )

                    with gr.Row():
                        user_input = gr.Textbox(
                            label="输入问题",
                            placeholder="请输入医疗问题（如「儿童发烧怎么办？」）或校园问题（如「如何申请宿舍？」）",
                            lines=3,
                            scale=8,
                            container=False
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                # 右侧功能区
                with gr.Column(scale=1):
                    # 角色选择
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 🎭 角色选择")
                        role = gr.Dropdown(
                            choices=["患者", "医生"],
                            value="患者",
                            label="回答风格",
                            info="患者：通俗语言 | 医生：专业术语",
                            container=False
                        )

                    # 参考资料
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 📚 参考资料")
                        reference = gr.Textbox(
                            label="资料来源",
                            placeholder="对话过程中会显示相关参考资料...",
                            lines=8,
                            interactive=False,
                            container=False
                        )

                    # 对话管理（仅保留TXT导出）
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 📤 对话管理")
                        clear_btn = gr.Button("清空对话", variant="secondary")
                        export_txt_btn = gr.Button("导出TXT", variant="secondary")
                        export_file = gr.File(
                            label="下载对话记录",
                            file_types=[".txt"],
                            interactive=False
                        )

        # 2.2 知识库可视化标签页
        with gr.TabItem("📖 知识库查看"):
            with gr.Row():
                # 医疗知识库
                with gr.Column(scale=1):
                    with gr.Column(elem_classes=["card", "dataframe-container"]):
                        gr.Markdown("### 🩺 医疗知识库")
                        df_medical = load_medical_knowledge_to_df()
                        medical_table = gr.Dataframe(
                            value=df_medical,
                            label="医疗知识列表",
                            interactive=False,
                            wrap=True
                        )
                        refresh_medical_btn = gr.Button("刷新医疗知识库", variant="secondary")

                # 校园知识库
                with gr.Column(scale=1):
                    with gr.Column(elem_classes=["card", "dataframe-container"]):
                        gr.Markdown("### 🏫 校园知识库")
                        df_campus = load_campus_knowledge_to_df()
                        campus_table = gr.Dataframe(
                            value=df_campus,
                            label="校园知识列表",
                            interactive=False,
                            wrap=True
                        )
                        refresh_campus_btn = gr.Button("刷新校园知识库", variant="secondary")

    # 3. 底部免责声明
    gr.Markdown("""
    <div class="footer">
        ⚠️ 免责声明：本系统仅提供信息参考，不构成医疗诊断建议，请遵医嘱 | 
        📅 更新时间：2024年
    </div>
    """)

    # ===================== 事件绑定 =====================
    # 核心对话逻辑
    submit_btn.click(
        fn=submit_msg,
        inputs=[user_input, role, chatbot],
        outputs=[chatbot, reference, user_input]
    )
    user_input.submit(
        fn=submit_msg,
        inputs=[user_input, role, chatbot],
        outputs=[chatbot, reference, user_input]
    )

    # 清空对话
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, reference, export_file, user_input]
    )

    # 仅保留TXT导出功能
    export_txt_btn.click(
        fn=export_chat_history_to_txt,
        inputs=chatbot,
        outputs=export_file
    )

    # 刷新知识库
    refresh_medical_btn.click(
        fn=load_medical_knowledge_to_df,
        outputs=medical_table
    )
    refresh_campus_btn.click(
        fn=load_campus_knowledge_to_df,
        outputs=campus_table
    )

# ===================== 运行程序 =====================
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=custom_css,
        theme=gr.themes.Soft()
    )