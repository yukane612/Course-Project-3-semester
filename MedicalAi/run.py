import subprocess
import sys


def main():
    print("正在启动医疗+校园双场景问答系统...")
    # 检查依赖
    required_packages = ["zhipuai", "gradio", "pymysql", "python-dotenv"]
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"缺少依赖包 {pkg}，正在安装...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

    # 启动Gradio界面
    from app_gradio import demo
    demo.launch(server_name="localhost", server_port=7860, share=False)


if __name__ == "__main__":
    main()