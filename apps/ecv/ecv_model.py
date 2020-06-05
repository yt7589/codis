#
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F

class EcvModel(torch.nn.Module):
    def __init__(self, dev, backbone='resnet50', num_classes=8999, 
                fc_dim=2048, fv_dim=128, fpn_fv_dim=32):
        super(EcvModel, self).__init__()
        self.name = 'apps.ecv.EcvModel'
        self.backbone = backbone
        self.num_classes = num_classes
        self.fc_dim = fc_dim
        self.fv_dim = fv_dim
        self.fpn_fv_dim = fpn_fv_dim
        self.base_dim = self.fc_dim - 4 * self.fv_dim
        self.raw_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.cnn_bone = torch.nn.Sequential(*list(self.raw_resnet50.children())[:-2])
        self.base_fc = torch.nn.Linear(self.fc_dim * 7 * 7, self.base_dim)
        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048], self.fpn_fv_dim)
        self.fv_fc = {
            'feat1': torch.nn.Linear(self.fpn_fv_dim*56*56, self.fv_dim),
            'feat2': torch.nn.Linear(self.fpn_fv_dim*28*28, self.fv_dim),
            'feat3': torch.nn.Linear(self.fpn_fv_dim*14*14, self.fv_dim),
            'feat4': torch.nn.Linear(self.fpn_fv_dim*7*7, self.fv_dim)
        }
        self.coeffs = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], requires_grad=True).to(dev)
        self.classifier = torch.nn.Linear(self.fc_dim, self.num_classes, bias=False)
        self.inter_layers = torchvision.models._utils.IntermediateLayerGetter(
            self.raw_resnet50, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )

    def forward(self, x):
        base_a1 = self.cnn_bone(x)
        base_a1 = torch.flatten(base_a_1, start_dim=1, end_dim=-1)
        base_a2 = self.base_fc(base_a1)
        fpn_a1 = self.inter_layers(x)
        fpn_y = self.fpn(fpn_a1)
        fv1 = self.get_feature_vector(self.fv_fc, fpn_y, 'feat1')
        fv2 = self.get_feature_vector(self.fv_fc, fpn_y, 'feat2')
        fv3 = self.get_feature_vector(self.fv_fc, fpn_y, 'feat3')
        fv4 = self.get_feature_vector(self.fv_fc, fpn_y, 'feat4')
        z = torch.cat([
            self.coeffs[0] * base_a2,
            self.coeffs[1] * fv1,
            self.coeffs[2] * fv2,
            self.coeffs[3] * fv3,
            self.coeffs[4] * fv4
        ], dim=1)
        return self.classifier(z)

