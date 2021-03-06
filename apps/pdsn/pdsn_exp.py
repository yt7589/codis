#
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torchvision
from ann.mask_rcnn import MaskRCNN
from ann.rpn import AnchorGenerator
import ann.resnet as resnet
from ann.backbone_utils import resnet_fpn_backbone

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
        # hyper parameter
        num_classes = 8999
        fc_dim = 2048
        feature_dim = 128
        base_dim = 2048 - 4 * feature_dim
        # network definition
        x = torch.rand(1, 3, 224, 224)
        m = torchvision.models.resnet50(pretrained=True)
        model = nn.Sequential(*list(m.children())[:-2])
        base_fc = nn.Linear(fc_dim * 7 * 7, base_dim)
        fv_fc = {
            'feat1': nn.Linear(32*56*56, feature_dim),
            'feat2': nn.Linear(32*28*28, feature_dim),
            'feat3': nn.Linear(32*14*14, feature_dim),
            'feat4': nn.Linear(32*7*7, feature_dim)
        }
        coefficients = {
            'base': 1.0,
            'feat1': 1.0,
            'feat2': 1.0,
            'feat3': 1.0,
            'feat4': 1.0
        }
        #avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        classifier = nn.Linear(2048, num_classes, bias=False)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            m, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )
        fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        # forward
        base_a_1 = model(x)
        base_a_1 = torch.flatten(base_a_1, start_dim=1, end_dim=-1)
        base_a_2 = base_fc(base_a_1)
        fpn_a1 = new_m(x)
        fvs = fpn(fpn_a1)
        fv1 = self.get_feature_vector(fv_fc, fvs, 'feat1')
        fv2 = self.get_feature_vector(fv_fc, fvs, 'feat2')
        fv3 = self.get_feature_vector(fv_fc, fvs, 'feat3')
        fv4 = self.get_feature_vector(fv_fc, fvs, 'feat4')
        z = torch.cat([
            coefficients['base']*base_a_2, 
            coefficients['feat1']*fv1,
            coefficients['feat2']*fv2,
            coefficients['feat3']*fv3,
            coefficients['feat4']*fv4
        ], dim = 1)
        y_hat = classifier(z)
        print('base_a_2: {0};'.format(base_a_2.shape))
        print('fv1: {0};'.format(fv1.shape))
        print('z: {0};'.format(z.shape))
        print('y_hat: {0};'.format(y_hat.shape))

    def get_feature_vector(self, fv_fc, fvs, feature_name):
        x_fv = fvs[feature_name]
        x_fv = torch.flatten(x_fv, start_dim=1, end_dim=-1)
        return fv_fc[feature_name](x_fv)

    def exp_mask_rcnn(self):
        # backbone = resnet.resnet50(pretrained=True)
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        #backbone.out_channels = 2048 # resnet 50
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                         output_size=7,
                                                         sampling_ratio=2)
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                              output_size=14,
                                                              sampling_ratio=2)
        model = MaskRCNN(backbone,
                          num_classes=2,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler,
                          mask_roi_pool=mask_roi_pooler)
        model = MaskRCNN(backbone,
                          num_classes=2,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler,
                          mask_roi_pool=mask_roi_pooler)
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        '''
        model.train()
        targets = [
            {
                'boxes': 
            }
        ]
        predictions = model(x, targets)
        '''
        predictions = model(x)
        print(predictions)

    def exp_(self):
        '''
        backbone = resnet_fpn_backbone('resnet50', True)
        x0 = np.array([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
        x = x0.from_numpy()
        backbone.eval()
        features = backbone(x)
        print(features)
        '''




























        