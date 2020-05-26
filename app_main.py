# 主程序入口
from apps.pdsn.pdsn_app import PdsnApp

def main():
    app = PdsnApp()
    app.startup()

if '__main__' == __name__:
    main()