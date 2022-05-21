import torch
import random
import numpy as np

from utils import transformsgpu
from models import get_deeplabv2
from utils import safe_add_params
from torch.nn import functional as F
from utils.visualize import visualize
from .basic import BasicModule, BasicDataModule


class SpatialMixDataModule(BasicDataModule):
    def __init__(self, params):
        super().__init__(params)


class SpatialMixModule(BasicModule):
    def __init__(self, params):
        super().__init__(params)
        self.ema_net = get_deeplabv2(num_classes=params.num_classes, restore_from=params.restore_from)
        for param in self.ema_net.parameters():
            param.requires_grad = False
            param.detach_()
        image_mean = torch.from_numpy(
            np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32).copy()
        )
        self.register_buffer("image_mean", image_mean)
    
    @staticmethod
    def add_params(params):
        BasicModule.add_params(params)
        safe_add_params(params, "--position-beta", type=float, default=0.5)
        return params
    
    def training_step(self, batch, batch_nb):
        loss = 0.0
        log_dict = {}

        image = batch["image"].float()
        # label_1 = batch["label"].long()
        with torch.no_grad():
            ema_pred = self.ema_net(image, size=image.shape[-2:])

        concat_image = []
        concat_ema_pred = []
        for image_idx in range(image.shape[0]):
            
            current_images = [image[image_idx, :, :, :].clone()]
            current_ema_preds = [ema_pred[image_idx, :, :, :].clone()]

            three_indexes = random.choices(list(set(list(range(image.shape[0]))) - {image_idx}), k=3)
            current_images.extend([image[idx, :, :, :].clone() for idx in three_indexes])
            current_ema_preds.extend([ema_pred[idx, :, :, :].clone() for idx in three_indexes])

            image_1 = current_images[0]
            image_2 = current_images[1]
            image_3 = current_images[2]
            image_4 = current_images[3]
            ema_pred_1 = current_ema_preds[0]
            ema_pred_2 = current_ema_preds[1]
            ema_pred_3 = current_ema_preds[2]
            ema_pred_4 = current_ema_preds[3]

            if self.params.position_beta > 0:
                width_position = 0.2 + np.random.beta(a=self.params.position_beta, b=self.params.position_beta) * 0.6
                start_width = int(width_position * image.shape[-1])
                height_position = 0.2 + np.random.beta(a=self.params.position_beta, b=self.params.position_beta) * 0.6
                start_height = int(height_position * image.shape[-2])
            else:
                start_width = int(0.5 * image.shape[-1])
                start_height = int(0.5 * image.shape[-2])


            # top left: 1, top right: 2
            image_12 = torch.cat(
                (
                    image_1[:, :start_height, :start_width],
                    image_2[:, :start_height, start_width:],
                ), 
                dim=-1
            )
            ema_pred_12 = torch.cat(
                (
                    ema_pred_1[:, :start_height, :start_width],
                    ema_pred_2[:, :start_height, start_width:],
                ), 
                dim=-1
            )
            # bottom left: 3, bottom right: 4
            image_34 = torch.cat(
                (
                    image_3[:, start_height:, :start_width],
                    image_4[:, start_height:, start_width:],
                ), 
                dim=-1
            )
            ema_pred_34 = torch.cat(
                (
                    ema_pred_3[:, start_height:, :start_width],
                    ema_pred_4[:, start_height:, start_width:],
                ), 
                dim=-1
            )
            concat_image.append(torch.cat([image_12, image_34], dim=-2))
            concat_ema_pred.append(torch.cat([ema_pred_12, ema_pred_34], dim=-2))

        concat_image = torch.stack(concat_image, dim=0)
        concat_ema_pred = torch.stack(concat_ema_pred, dim=0)

        if np.random.uniform() < 0.5:
            concat_image = torch.flip(concat_image, dims=[-1])
            concat_ema_pred = torch.flip(concat_ema_pred, dims=[-1])
        concat_conf, concat_label = concat_ema_pred.max(dim=1)

        aug_cat_image, _ = transformsgpu.color_jitter(
            colorJitter=np.random.uniform(), 
            img_mean=self.image_mean, 
            data=concat_image, 
            target=None
        )
        aug_cat_image, _ = transformsgpu.gaussian_blur(
            blur=np.random.uniform(),
            data=aug_cat_image, 
            target=None
        )

        unlabeled_weight = torch.sum(concat_conf.ge(0.968).long() == 1).item() / np.size(np.array(concat_label.cpu()))
        pixel_weight = unlabeled_weight * torch.ones_like(concat_conf)

        aug_out = self(aug_cat_image, size=concat_label.shape[-2:])
        loss_consistency = F.cross_entropy(aug_out, concat_label, reduction="none")
        loss_consistency = torch.mean(pixel_weight * loss_consistency)
        log_dict["loss/consistency"] = loss_consistency.item()
        loss += loss_consistency

        alpha_teacher = min(1 - 1 / (self.global_step + 1), 0.99)
        for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

        if self.params.visualize_interval > 0 and self.global_step % self.params.visualize_interval == 0 and self.local_rank == 0:
            visualize(aug_cat_image, concat_label, aug_out.argmax(dim=1), "./with_aug.png")
        
        self.log_dict(log_dict, prog_bar=True, logger=True)
        return loss
