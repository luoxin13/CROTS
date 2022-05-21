# Basic Module for running the experiment: 
#     adapting Black-box Source-only Model onto Cityscapes Datasets

import os
import torch
import pytorch_lightning as pl

from utils import get_iou_string
from models import get_deeplabv2
from utils import safe_add_params
from torch.nn import functional as F
from utils.visualize import visualize
from datasets import CityscapesDataSet
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from utils.confusion_matrix import get_iou


class BasicDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.val_batch_size = params.val_batch_size
        self.num_workers = params.num_workers
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.set_datasets()
    
    @staticmethod
    def add_params(params):
        safe_add_params(params, "--image-root", type=str, default="./data/Cityscapes/leftImg8bit")
        safe_add_params(params, "--label-root", type=str, default="./data/Cityscapes/gtFine2")
        safe_add_params(params, "--list-root", type=str, default="./datasets/list/cityscapes")
        safe_add_params(params, "--ignore-index", type=int, default=255)
        safe_add_params(params, "--train-batch-size", type=int, default=4, help="size of the batches")
        safe_add_params(params, "--test-batch-size", type=int, default=2, help="size of the batches")
        safe_add_params(params, "--val-batch-size", type=int, default=2, help="size of the batches")
        safe_add_params(params, "--num-workers", type=int, default=8, help="size of workers")
        return params

    def set_datasets(self):
        # ATTETENTION: This can be modified according to your need
        self.train_dataset = CityscapesDataSet(
            image_root=os.path.join(self.params.image_root, "train"),
            label_root=os.path.join(self.params.label_root, "train"),
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
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class BasicModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.learning_rate = params.lr
        self.ignore_index = params.ignore_index
        self.num_classes = params.num_classes
        self.confusion_matrix = ConfusionMatrix(num_classes=params.num_classes+1)
        self.net = get_deeplabv2(num_classes=params.num_classes, restore_from=params.restore_from)
        self.miou_record = os.path.join(
            params.log_tb_dir, 
            params.module_name,
            params.experiment_name, 
            "miou_record.csv"
        )
        os.makedirs(os.path.dirname(self.miou_record), exist_ok=True)
        self.class_names = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "veg",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "mbike",
            18: "bike",
        }
        self.debug_title = False
        self.write_val = False # skip the debugging val results

    @staticmethod
    def add_params(params):
        safe_add_params(params, "--num-classes", type=int, default=19)
        safe_add_params(params, "--restore-from", type=str, default="")
        safe_add_params(params, "--optimizer", type=str, default="adam")
        safe_add_params(params, "--lr", type=float, default=1.0e-4, help="learning rate")
        safe_add_params(params, "--log-tb-dir", type=str, default="./experiments")
        safe_add_params(params, "--experiment-name", type=str, default="experiment")
        safe_add_params(params, "--visualize-interval", type=int, default=50)
        safe_add_params(params, "--flip-when-test", action="store_true")
        safe_add_params(params, "--ignore-index", type=int, default=255)
        return params

    def configure_optimizers(self):
        # ATTETENTION: This can be modified according to your need
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        if self.params.optimizer == "SGD":
            opt = torch.optim.SGD(
                self.net.optim_parameters(learning_rate=self.learning_rate),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=5.0e-4,
                nesterov=True,
            )
            sch = torch.optim.lr_scheduler.LambdaLR()
        return [opt], [sch]
        
    def forward(self, x, size=None):
        return self.net(x, size=size)

    def training_step(self, batch, batch_nb):
        # ATTETENTION: This can be modified according to your need

        loss = 0.0
        log_dict = {}

        image = batch["image"].float()
        label = batch["label"].long()
        out = self(image, size=label.shape[-2:])
        loss_seg = F.cross_entropy(out, label, ignore_index=self.ignore_index)
        loss += loss_seg
        log_dict["seg"] = loss_seg.item()
        
        # aug_image = batch["aug_image"].float()
        # aug_label = batch["aug_label"].long()
        # aug_out = self(aug_image, size=aug_label.shape[-2:])
        # loss_aug_seg = F.cross_entropy(aug_out, aug_label, ignore_index=self.ignore_index)
        # loss += loss_aug_seg
        # log_dict["aug_seg"] = loss_aug_seg.item()

        # FOR DEBUGGING ONLY
        if self.global_step % self.params.visualize_interval == 0 and self.local_rank == 0:
            visualize(batch["image"], label, out.argmax(dim=1), "./without_aug.png")
            # visualize(batch["aug_image"], aug_label, aug_out.argmax(dim=1), "./with_aug.png")

        self.log_dict(log_dict, prog_bar=False, logger=True)
        return loss
    
    def on_test_start(self):
        self.confusion_matrix.reset()
        
    def test_step(self, batch, batch_idx):
        image = batch["image"].float()
        label = batch["label"].long()
        out = self(image, size=label.shape[-2:])
        if self.params.flip_when_test:
            out += torch.flip(self(torch.flip(image, dims=[-1]), size=label.shape[-2:]), dims=[-1])
        out = torch.argmax(F.softmax(out, dim=1), dim=1)
        label[label > self.num_classes] = self.num_classes
        self.confusion_matrix.update(out, label)
    
    def test_epoch_end(self, outputs):
        iou_classes = get_iou(self.confusion_matrix.compute())
        mean_iou = torch.nanmean(iou_classes).item()
        self.confusion_matrix.reset()
        self.log("test_miou", mean_iou * 100, prog_bar=True, logger=True)
        iou_str = get_iou_string(iou_classes, self.params.num_classes)
        if self.local_rank == 0:
            with open(self.miou_record, "a") as fi:
                fi.write(f"{os.path.basename(self.params.restore_from)}-flip={self.params.flip_when_test}, {iou_str}\n")
    
    def on_validation_start(self):
        self.confusion_matrix.reset()
        if not self.debug_title:
            iou_title_str = ' , '.join([self.class_names[i] for i in range(self.params.num_classes)] + ['Mean', 'Mean-16', 'Mean-13'])
            if self.local_rank == 0:
                with open(self.miou_record, "a") as fi:
                    fi.write(f"ClassNames , {iou_title_str}\n")
            self.debug_title = True
    
    def validation_step(self, batch, batch_idx):
        image = batch["image"].float()
        label = batch["label"].long()
        out = self(image, size=label.shape[-2:])
        out = torch.argmax(F.softmax(out, dim=1), dim=1)
        label[label > self.num_classes] = self.num_classes
        self.confusion_matrix.update(out, label)

    def validation_epoch_end(self, outputs):
        iou_classes = get_iou(self.confusion_matrix.compute())
        mean_iou = torch.nanmean(iou_classes).item()
        self.confusion_matrix.reset()
        self.log("val_miou", mean_iou * 100, prog_bar=True, logger=True)
        if self.write_val:
            iou_str = get_iou_string(iou_classes, self.params.num_classes)
            if self.local_rank == 0:
                with open(self.miou_record, "a") as fi:
                    fi.write(f"step-{self.global_step + 1}, {iou_str}\n")
        else:
            self.write_val = True
