# PennFudan行人检测和分割数据集应用，综合应用分类、检测、实例分割应用
import torch
from apps.pdsn.pnfn_dataset import PnfnDataset
from torchvision import utils
from apps.pdsn.pnfn_dataset import PnfnDataset
from apps.pdsn.pdsn_model import PdsnModel
from apps.pdsn.engine import train_one_epoch, evaluate
#
from apps.pdsn.pdsn_exp import PdsnExp

class PdsnApp(object):
    def __init__(self):
        self.name = 'apps.pdsn.PdsnApp'

    def startup(self):
        print('分类检测实例分割综合实验平台 v0.0.1 自研mask_rcnn.py')
        i_debug = 1
        if 1 == i_debug:
            exp = PdsnExp()
            exp.exp_()
            return
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # our dataset has two classes only - background and person
        num_classes = 2
        # use our dataset and defined transformations
        dataset = PnfnDataset('./datasets/PennFudanPed', PnfnDataset.get_transform(train=True))
        dataset_test = PnfnDataset('./datasets/PennFudanPed', PnfnDataset.get_transform(train=False))
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=PnfnDataset.collate_fn)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=PnfnDataset.collate_fn)
        model = PdsnModel.get_model_instance_segmentation(num_classes)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        num_epochs = 10
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)
        print('^_^')
