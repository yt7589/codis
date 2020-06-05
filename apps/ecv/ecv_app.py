#
from apps.ecv.ecv_model import EcvModel

class EcvApp(object):
    def __init__(self):
        self.name = 'apps.ecv.EcvApp'

    def startup(self):
        print('测试平台')
        model = EcvModel()
        y_hat = model(x)
        print('y_hat: {0};'.format(y_hat))