import os
import cv2
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from models import get_deeplabv2
from matplotlib import pyplot as plt
from utils.visualize import visualize
from datasets import CityscapesDataSet
from torch.utils import data, model_zoo
from torchmetrics import ConfusionMatrix
from utils.patch_pool import PatchPool
from utils.confusion_matrix import get_acc_and_iou


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


DATA_DIRECTORY = './data/Cityscapes'
DATA_LIST_PATH = './datasets/list/cityscapes/train.txt'
SAVE_PATH = './experiments/hard_patch_from_src_model/'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500  # Number of images in the validation set.
RESTORE_FROM = './pretrained/gta5_only.pth'


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_arguments()

    model = get_deeplabv2(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from, map_location="cpu")
    else:
        saved_state_dict = torch.load(args.restore_from, map_location="cpu")
    model.load_state_dict(saved_state_dict)

    model.eval().cuda(args.gpu)

    test_dataloader = data.DataLoader(
        CityscapesDataSet(
            os.path.join(args.data_dir, "leftImg8bit/train"),
            os.path.join(args.data_dir, "gtFine2/train"), 
            args.data_list, 
            image_size=(1024, 512),
            label_size=(1024, 512),
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    # class_pixel_counts = np.zeros(args.num_classes, dtype=np.float64)
    # gt_class_pixel_counts = np.zeros(args.num_classes, dtype=np.float64)
    # confidence_counts = np.zeros(args.num_classes, dtype=np.float64)
    # confidence_weights = np.zeros(args.num_classes, dtype=np.float64)

    # cm = ConfusionMatrix(num_classes=args.num_classes + 1).cuda(args.gpu)
    # cm.reset()

    # for sample_index, batches in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    #     if sample_index > 100: break
    #     batch = batches
    #     image = batch["image"].cuda(args.gpu)
    #     label = batch["label"].cuda(args.gpu).long()
    #     label[label > args.num_classes] = args.num_classes
    #     pred = model(image, size=image.shape[-2:])
    #     # pred += torch.flip(model(torch.flip(image, dims=[-1]), size=image.shape[-2:]), dims=[-1])
    #     pred_conf, pred_label = pred.max(dim=1)
    #     cm.update(pred_label, label)
    #     for class_idx in range(args.num_classes):
    #         class_pixel_counts[class_idx] += (pred_label == class_idx).sum().item()
    #         gt_class_pixel_counts[class_idx] += (label == class_idx).sum().item()
    #         confidence_counts[class_idx] += pred_conf[pred_label == class_idx].sum().item()
    #         confidence_weights[class_idx] += (pred_label == class_idx).sum().item()
    
    # acc, iou = get_acc_and_iou(cm.compute(), drop_last=True)
    # acc = acc.cpu().numpy()
    # iou = iou.cpu().numpy()
    # confidence = confidence_counts / (confidence_weights + 1e-8)

    # print("     acc:\t", np.argsort(acc))
    # print("     iou:\t", np.argsort(iou))
    # print("    conf:\t", np.argsort(confidence))
    # print("pred_cnt:\t", np.argsort(class_pixel_counts))
    # print("  gt_cnt:\t", np.argsort(gt_class_pixel_counts))

    # rare_classes = list(set(np.argsort(confidence)[:5].tolist() + np.argsort(class_pixel_counts)[:5].tolist()))
    # print("rare_classes: ", rare_classes)

    # rare_classes = {6, 7, 12, 16, 17, 18} # GTA5
    rare_classes = {6, 7, 12, 17, 18} # Synthia

    save_image_root = os.path.join(
        args.save, 
        "patches", 
    )
    save_label_root = os.path.join(
        args.save, 
        "labels", 
    )
    save_color_label_root = os.path.join(
        args.save, 
        "color_labels", 
    )
    os.makedirs(save_image_root, exist_ok=True)
    os.makedirs(save_label_root, exist_ok=True)
    os.makedirs(save_color_label_root, exist_ok=True)


    num_samples = {i:0 for i in range(args.num_classes)}
    with torch.no_grad():
        dataloader = test_dataloader
        for sample_index, batches in tqdm(enumerate(dataloader), total=len(test_dataloader)):
            # if sample_index > 300:
            #     break
            batch = batches
            image = batch["image"].cuda(args.gpu)
            label = batch["label"].cuda(args.gpu).long()
            pred = model(image, size=image.shape[-2:])
            pred += torch.flip(model(torch.flip(image, dims=[-1]), size=image.shape[-2:]), dims=[-1])
            pred_label = pred.argmax(dim=1)
            for batch_idx in range(image.shape[0]):
                image_i = image[batch_idx]
                pred_label_i = pred_label[batch_idx]
                name_i = batch["name"][batch_idx]
                step_height = int(image_i.shape[-2] / 4)
                step_width = int(image_i.shape[-1] / 4)
                patch_idx = 0
                for i in range(4):
                    start_height = step_height * i
                    for j in range(4):
                        start_width = step_width * j
                        image_patch = image_i[:, start_height:start_height+step_height, start_width:start_width+step_width]
                        label_patch = pred_label_i[start_height:start_height+step_height, start_width:start_width+step_width]
                        patch_idx += 1
                        inter = set.intersection(
                            set(torch.unique(label_patch).cpu().numpy().tolist()),
                            rare_classes
                        )
                        if len(inter) > 0:
                            image_patch_np = image_patch.permute(1, 2, 0).cpu().numpy()
                            image_patch_np += IMG_MEAN
                            image_patch_np = image_patch_np[:, :, ::-1]
                            image_patch_np = image_patch_np.astype(np.uint8)
                            label_patch_np = label_patch.cpu().numpy().astype(np.uint8)
                            save_image_path = os.path.join(
                                args.save, 
                                "patches", 
                                os.path.basename(name_i).replace(".png", f"_{patch_idx}.png")
                            )
                            save_label_path = os.path.join(
                                args.save, 
                                "labels", 
                                os.path.basename(name_i).replace(".png", f"_{patch_idx}.png")
                            )
                            save_color_label_path = os.path.join(
                                args.save, 
                                "color_labels", 
                                os.path.basename(name_i).replace(".png", f"_{patch_idx}.png")
                            )
                            Image.fromarray(image_patch_np).save(save_image_path)
                            Image.fromarray(label_patch_np).save(save_label_path)
                            colorize_mask(label_patch_np).save(save_color_label_path)


if __name__ == '__main__':
    main()
