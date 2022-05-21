import os
import math
import random
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision.transforms import ColorJitter


class CityscapesDataSet(data.Dataset):
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    def __init__(
        self, 
        image_root, 
        label_root,
        list_path,
        image_size=(1024, 512),
        label_size=None,
        augment=None,
        max_iters=None,
        shuffle=False,
        plabel_root="",
    ):
        self.list_path = list_path
        self.image_root = image_root
        self.label_root = label_root
        self.plabel_root = plabel_root
        self.image_size = image_size
        self.label_size = label_size
        self.augment = augment
        with open(list_path) as fi:
            image_names = [i_id.strip().split(' ')[0] for i_id in fi]
        if max_iters is not None and isinstance(max_iters, int):
            image_names = image_names * (math.ceil(max_iters / len(image_names)))
            image_names = image_names[: max_iters + 1]
        if shuffle:
            random.shuffle(image_names)
            
        self.files = []
        for name in image_names:
            img_file = os.path.join(self.image_root, name)
            if self.plabel_root != "":
                lab_file = os.path.join(self.plabel_root, name.replace('leftImg8bit', 'gtFine_labelIds'))
            else:
                lab_file = os.path.join(self.label_root, name.replace('leftImg8bit', 'gtFine_labelIds'))
            self.files.append({
                "img": img_file,
                'lab': lab_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def transform_image(self, image):
        image = np.array(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        #
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["lab"])
        # resize
        image = image.resize(self.image_size, Image.BICUBIC)
        if self.label_size is not None:
            label = label.resize(self.label_size, Image.NEAREST)
        #
        aug_image = ColorJitter(0.5, 0.5, 0.5, 0.5)(image)
        if self.augment is not None:
            aug_image = self.augment(image)
        aug_label = label.copy()
        #
        image = self.transform_image(image).copy()
        label = np.array(label, np.int64).copy()
        aug_image = self.transform_image(aug_image).copy()
        aug_label = np.array(aug_label, np.int64).copy()
        #
        return dict(
            image=image,
            label=label,
            aug_image=aug_image,
            aug_label=aug_label,
            name=datafiles["name"],
        )
