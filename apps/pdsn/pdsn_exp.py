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
        for k, v in x0.items():
            print('{0}:{1};'.format(k, v.shape))
        '''
        x = OrderedDict()
        x['feat0'] = torch.rand(1, 10, 64, 64)
        x['feat2'] = torch.rand(1, 20, 16, 16)
        x['feat3'] = torch.rand(1, 30, 8, 8)
        '''
        output = fpn(x0)
        print([(k, v.shape) for k, v in output.items()])
        