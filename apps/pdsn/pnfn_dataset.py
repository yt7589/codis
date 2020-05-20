# PennFudan行人检测分割数据集
#
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import re
import cv2
import apps.pdsn.transforms as T


class PnfnDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def get_image_mask(self):
        ann_path = '{0}/Annotation'.format(self.root)
        imgs = []
        for ann_file in sorted(os.listdir(ann_path)):
            imgs.append('{0}/PNGImages/{1}.png'.format(self.root, ann_file[:-4]))
        img_obj = Image.open(imgs[5])
        plt.figure("dog")
        plt.imshow(img_obj)
        plt.show()
        img_file = imgs[5].split('/')
        mask_img = Image.open('{0}/PedMasks/{1}_mask.png'.format(self.root, img_file[-1][:-4]))
        mask_img.putpalette([
            0, 0, 0, # black background
            255, 0, 0, # index 1 is red
            255, 255, 0 # index 2 is yellow
        ])
        plt.figure('mask')
        plt.imshow(mask_img)
        plt.show()

    def get_detection_boxes(self):
        ann_path = '{0}/Annotation'.format(self.root)
        anns = []
        for ann_file in sorted(os.listdir(ann_path)):
            anns.append('{0}/Annotation/{1}'.format(self.root, ann_file))
            print('{0}/Annotation/{1}'.format(self.root, ann_file))
        with open(anns[5], 'r', encoding='utf-8') as fd:
            arrs = anns[5].split('/')
            img = cv2.imread('{0}/PNGImages/{1}.png'.format(self.root, arrs[-1][:-4]))
            for line in fd:
                print(line)
                if re.findall('Xmin', line):
                    pt = [int(x) for x in re.findall(r'\d+',line)]
                    box_img = img[pt[2]:pt[4], pt[1]:pt[3]]
                    plt.figure('box')
                    plt.imshow(box_img)
                    plt.show()


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))