import os

from utils import safe_add_params
from datasets import CityscapesDataSet
from .basic import BasicModule, BasicDataModule


class BaselineDataModule(BasicDataModule):
    def __init__(self, params):
        super().__init__(params)
    
    @staticmethod
    def add_params(params):
        BasicDataModule.add_params(params)
        safe_add_params(params, "--pseudo-label-root", type=str, default="./experiments/source_only_pseudo_label/label_ids")
        return params

    def set_datasets(self):
        # MODIFIED: Using Pseudo Label Root
        self.train_dataset = CityscapesDataSet(
            image_root=os.path.join(self.params.image_root, "train"),
            label_root=os.path.join(self.params.label_root, "train"),
            plabel_root=os.path.join(self.params.pseudo_label_root),
            list_path=os.path.join(self.params.list_root, "train.txt"),
            image_size=(1024, 512),
            label_size=(1024, 512),
        )
        self.test_dataset = CityscapesDataSet(
            image_root=os.path.join(self.params.image_root, "val"),
            label_root=os.path.join(self.params.label_root, "val"),
            list_path=os.path.join(self.params.list_root, "val.txt"),
            image_size=(1024, 512),
            augment=None,
        )
        self.val_dataset = CityscapesDataSet(
            image_root=os.path.join(self.params.image_root, "val"),
            label_root=os.path.join(self.params.label_root, "val"),
            list_path=os.path.join(self.params.list_root, "val.txt"),
            image_size=(1024, 512),
        )


class BaselineModule(BasicModule):
    def __init__(self, params):
        super().__init__(params)

    # The same codes as those in BasicModule
    # def training_step(self, batch, batch_nb):
