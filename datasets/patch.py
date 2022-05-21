import os
import math
import numpy as np

from glob import glob
from PIL import Image
from torch.utils import data


class PatchDataset(data.Dataset):
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    def __init__(
        self,
        root, 
        image_size=(256, 128),
        label_size=None,
        max_iters=None,
    ):
        self.root = root
        self.image_size = image_size
        self.label_size = label_size
        
        image_names = sorted(glob(os.path.join(self.root, "patches", "*.png")))
        if max_iters is not None and isinstance(max_iters, int):
            image_names = image_names * (math.ceil(max_iters / len(image_names)))
            image_names = image_names[: max_iters + 1]
        label_names = map(lambda x: x.replace("/patches/", "/labels/"), image_names)
        self.files = list(zip(image_names, label_names))
        
    def __len__(self):
        return len(self.files)

    def transform_image(self, image):
        image = np.array(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= self.IMG_MEAN
        image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, index):
        image_path, label_path = self.files[index]
        #
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)
        # resize
        image = image.resize(self.image_size, Image.BICUBIC)
        if self.label_size is not None:
            label = label.resize(self.label_size, Image.NEAREST)
        #
        image = self.transform_image(image).copy()
        label = np.array(label, np.int64).copy()
        #
        return dict(
            image=image,
            label=label,
        )
