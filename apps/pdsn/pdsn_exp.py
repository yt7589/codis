#
from collections import OrderedDict
import torch
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
        m = torchvision.models.resnet50(pretrained=True)
        new_m = torchvision.models._utils.IntermediateLayerGetter(
            m, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )
        x0 = new_m(torch.rand(1, 3, 224, 224))
        print('Features:')
        for k, v in x0.items():
            print('{0}:{1};'.format(k, v.shape))
        fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], 32)
        output = fpn(x0)
        print('FPNs:')
        for k, v in output.items():
            print('{0}:{1};'.format(k, v.shape))

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
        backbone = resnet_fpn_backbone('resnet50', True)
        x0 = np.array([torch.rand(3, 300, 400), torch.rand(3, 500, 400)])
        x = x0.from_numpy()
        backbone.eval()
        features = backbone(x)
        print(features)




























        