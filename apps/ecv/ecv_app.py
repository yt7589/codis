#
import torch
from apps.ecv.ecv_model import EcvModel
from apps.ecv.ecv_config import LoadConfig, load_data_transformers
from apps.ecv.ecv_dataset import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, EcvDataset

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='CUB', type=str)
    parser.add_argument('--save', dest='resume',
                        default=None,
                        type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume',
                        action='store_true')
    parser.add_argument('--epoch', dest='epoch',
                        default=360, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=8, type=int)
    parser.add_argument('--vb', dest='val_batch',
                        default=512, type=int)
    parser.add_argument('--sp', dest='save_point',
                        default=5000, type=int)
    parser.add_argument('--cp', dest='check_point',
                        default=5000, type=int)
    parser.add_argument('--lr', dest='base_lr',
                        default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step',
                        default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio',
                        default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers',
                        default=16, type=int)
    parser.add_argument('--vnw', dest='val_num_workers',
                        default=32, type=int)
    parser.add_argument('--detail', dest='discribe',
                        default='', type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=448, type=int)
    parser.add_argument('--cls_2', dest='cls_2',
                        action='store_true')
    parser.add_argument('--cls_mul', dest='cls_mul',
                        action='store_true')
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

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
        # 测试数据集的使用
        args = parse_args()
        config = LoadConfig(args, 'train')
        config.cls_2 = args.cls_2
        config.cls_2xmul = args.cls_mul
        transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)
        # inital dataloader
        train_set = dataset(Config = config,\
                            anno = config.train_anno,\
                            common_aug = transformers["common_aug"],\
                            swap = transformers["swap"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["train_totensor"],\
                            train = True)

        trainval_set = dataset(config = config,\
                            anno = config.train_anno,\
                            common_aug = transformers["None"],\
                            swap = transformers["None"],\
                            swap_size=args.swap_num, \
                            totensor = transformers["val_totensor"],\
                            train = False,
                            train_val = True)

        val_set = dataset(config = config,\
                        anno = config.val_anno,\
                        common_aug = transformers["None"],\
                        swap = transformers["None"],\
                            swap_size=args.swap_num, \
                        totensor = transformers["test_totensor"],\
                        test=True)

        dataloader = {}
        dataloader['train'] = torch.utils.data.DataLoader(train_set,\
                                                    batch_size=args.train_batch,\
                                                    shuffle=True,\
                                                    num_workers=args.train_num_workers,\
                                                    collate_fn=collate_fn4train if not config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if config.use_backbone else False,
                                                    pin_memory=True)

        setattr(dataloader['train'], 'total_item_len', len(train_set))

        dataloader['trainval'] = torch.utils.data.DataLoader(trainval_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4val if not config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if config.use_backbone else False,
                                                    pin_memory=True)

        setattr(dataloader['trainval'], 'total_item_len', len(trainval_set))
        setattr(dataloader['trainval'], 'num_cls', config.numcls)

        dataloader['val'] = torch.utils.data.DataLoader(val_set,\
                                                    batch_size=args.val_batch,\
                                                    shuffle=False,\
                                                    num_workers=args.val_num_workers,\
                                                    collate_fn=collate_fn4test if not config.use_backbone else collate_fn4backbone,
                                                    drop_last=True if config.use_backbone else False,
                                                    pin_memory=True)

        setattr(dataloader['val'], 'total_item_len', len(val_set))
        setattr(dataloader['val'], 'num_cls', config.numcls)
