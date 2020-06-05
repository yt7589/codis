#
import torch
from apps.ecv.ecv_model import EcvModel

class EcvApp(object):
    def __init__(self):
        self.name = 'apps.ecv.EcvApp'

    def startup(self):
        print('测试平台')
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = EcvModel(dev)
        x = torch.rand(1, 3, 224, 224)
        y_hat = model(x)
        print('y_hat: {0};'.format(y_hat.shape))