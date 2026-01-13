import gradio as  gr



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            gr.Textbox(label="姓名", placeholder="请输入您的姓名")
            with gr.Column(scale=1):
                gr.Slider(label="声音", minimum=0, maximum=100, value=50, step=2, interactive=True)


demo.launch()