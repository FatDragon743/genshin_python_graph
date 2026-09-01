"""
原神抽卡欧非分析 — 入口（本地网页）

用法: python test3.py
浏览器打开 http://127.0.0.1:8765/
"""

from wish_app.webapp import run


def main():
    run()


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"错误：缺少必要的库 - {e}")
        print("请执行: pip install -r requirements.txt")
