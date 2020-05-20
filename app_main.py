# 主程序入口
from apps.pdsn.pdsn_app import PdsnApp

def main():
    print('分类检测实例分割综合实验平台 v0.0.1')
    app = PdsnApp()
    app.startup()

if '__main__' == __name__:
    main()