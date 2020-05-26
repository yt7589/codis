#
from collections import OrderedDict
import torch
import torchvision

class PdsnExp(object):
    def __init__(self):
        self.name = ''

    def exp_intermediate_layer_getter(self):
        m = torchvision.models.resnet50(pretrained=True)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            m, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )
        out = new_m(torch.rand(1, 3, 224, 224))
        print([(k, v.shape) for k, v in out.items()])

    def exp_feature_pyramid_network(self):
        '''
        out = fpn(x)
        for k, v in out.items():
            print('{0}:{1};'.format(k, v.shape))
        '''
        fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        # fpn = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        m = torchvision.models.resnet50(pretrained=True)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            m, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )
        x0 = new_m(torch.rand(1, 3, 224, 224))
        print('Features:')
        for k, v in x0.items():
            print('{0}:{1};'.format(k, v.shape))
        output = fpn(x0)
        print('FPNs:')
        for k, v in output.items():
            print('{0}:{1};'.format(k, v.shape))
        