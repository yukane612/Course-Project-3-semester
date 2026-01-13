import gradio as gr


def age_fn(value):
    num = float(value)
    return str(num * 2)

def file_fn(value):
    return  value

def show_fn(move):
    return f"滑块上面的数据是:{move}"

def submit_fn(msg):
    return f"您说的是:{msg}"

with gr.Blocks() as demo:
    #click事件
    age = gr.Textbox(label="年龄",placeholder="请输入年龄")

    btn = gr.Button("提交")

    show1 = gr.Textbox(label="年龄的两倍")

    btn.click(fn=age_fn,inputs=age,outputs=show1)

    #change事件
    move = gr.Slider(minimum=20,maximum=50,value=30,interactive=True)
    show2 = gr.Textbox(label="显示滑块数据")

    move.change(fn = show_fn,inputs = move,outputs = show2)

    #upload事件

    file = gr.File(label="请上传文件")

    show3 = gr.Textbox(label="上传的文件名")

    file.upload(fn=file_fn,inputs=file,outputs=show3)

    #submit事件
    gr.Markdown("###<center>标题三</center>")

    show4 = gr.Textbox(label="这里")
    user_input = gr.Textbox

demo.launch()