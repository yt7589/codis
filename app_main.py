# 主程序入口
from apps.pdsn.pdsn_app import PdsnApp
from apps.ecv.ecv_app import EcvApp

def main():
    #app = PdsnApp()
    app = EcvApp()
    app.startup()

if '__main__' == __name__:
    main()