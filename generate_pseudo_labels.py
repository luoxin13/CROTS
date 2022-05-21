import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from PIL import Image
from models import get_deeplabv2
from datasets import CityscapesDataSet
from torch.utils import data, model_zoo


DATA_DIRECTORY = './data/Cityscapes'
DATA_LIST_PATH = './datasets/list/cityscapes/train.txt'
SAVE_PATH = './experiments/gta5_pseudo_label/'

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
    parser.add_argument("--multiscale", action="store_true",
                        help="whether to ensemble multiscale.")
    return parser.parse_args()


def main():
    args = get_arguments()
    
    pseudo_label_path = os.path.join(args.save, "label_ids", "train")
    color_pseudo_label_path = os.path.join(args.save, "label_color", "train")

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
            image_size=(1024, 512)
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    entropy_values = {}

    with torch.no_grad():
        dataloader = test_dataloader
        for sample_index, batches in tqdm(enumerate(dataloader), total=len(test_dataloader)):
            batch = batches
            image = batch["image"].cuda(args.gpu)
            pred = model(image, size=image.shape[-2:])
            pred += torch.flip(model(torch.flip(image, dims=[-1]), size=image.shape[-2:]), dims=[-1])
            label = pred.argmax(dim=1)
            entropy = -pred.softmax(dim=1) * pred.log_softmax(dim=1)
            entropy = entropy.sum(dim=1)
            for batch_index in range(image.shape[0]):
                name_i = batch["name"][batch_index]
                label_i = label[batch_index].cpu().numpy().astype(np.uint8)
                label_i_image = Image.fromarray(label_i)
                label_i_color_image = colorize_mask(label_i)
                entropy_values[name_i] = entropy[batch_index].mean()
                # print(f"{name_i}: {entropy[batch_index].mean().item()}")
                
                save_path_id = os.path.join(
                    pseudo_label_path, 
                    name_i.replace('leftImg8bit', 'gtFine_labelIds')
                )
                save_path_color = os.path.join(
                    color_pseudo_label_path, 
                    name_i.replace('leftImg8bit', 'gtFine_labelIds')
                )
                os.makedirs(os.path.dirname(save_path_id), exist_ok=True)
                label_i_image.save(save_path_id)
                os.makedirs(os.path.dirname(save_path_color), exist_ok=True)
                label_i_color_image.save(save_path_color)
                
    name_entropy_list = sorted(entropy_values.items(), key=lambda x: x[1])
    
    half_len = len(name_entropy_list) // 2
    easy_list = [name[0]+' '+str(name[1].item())+'\n' for name in name_entropy_list[:half_len]]
    hard_list = [name[0]+' '+str(name[1].item())+'\n' for name in name_entropy_list[half_len:]]
    
    with open(os.path.join(args.save, "easy.txt"), "w") as fi:
        fi.writelines(easy_list)
    with open(os.path.join(args.save, "hard.txt"), "w") as fi:
        fi.writelines(hard_list) 
                

if __name__ == '__main__':
    main()
