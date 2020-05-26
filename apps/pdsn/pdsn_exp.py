#
import torch
import torchvision

class PdsnExp(object):
    def __init__(self):
        self.name = ''

    def exp_intermediate_layer_getter(self):
        m = torchvision.models.resnet18(pretrained=True)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            m, {'layer1': 'feat1', 'layer3': 'feat2'}
        )
        out = new_m(torch.rand(1, 3, 224, 224))
        print([(k, v.shape) for k, v in out.items()])
        