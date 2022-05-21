import numpy as np

from PIL import Image
from matplotlib import pyplot as plt



def norm(x, bgr_rgb=True):
    if len(x.shape) == 3 and bgr_rgb:
        if x.shape[0] == 3:
            x = x[::-1, :, :]
        elif x.shape[2] == 3:
            x = x[:, :, ::-1] 
    return (x - x.min()) / (x.max() - x.min())


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((20, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  colormap[19] = [0,0,0] # void class
  return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))
    if np.max(label) >= 256:
        raise ValueError(
            'label value too large: {} >= {}.'.format( np.max(label), 256)
        )
    label[label >= 19] = 19
    colormap = create_cityscapes_label_colormap()
    return colormap[label]


def visualize(image, label, pred_label, path):
    if len(image.shape) == 4:
        image = image[0]
        label = label[0]
        pred_label = pred_label[0]
    
    if not isinstance(image, np.ndarray):
        image = image.cpu().numpy().astype(np.float32)
    if not isinstance(label, np.ndarray):
        label = label.cpu().numpy().astype(np.uint8)
    if not isinstance(pred_label, np.ndarray):
        pred_label = pred_label.cpu().numpy().astype(np.uint8)
        
    image = np.transpose(image, [1, 2, 0])
        
    pil_image = Image.fromarray((norm(image) * 255.0).astype(np.uint8))
    pil_label = Image.fromarray(label_to_color_image(label))
    pil_pred_label = Image.fromarray(label_to_color_image(pred_label))

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].set(xticks=[], yticks=[])
    ax[0].imshow(pil_image)
    ax[1].set(xticks=[], yticks=[])
    ax[1].imshow(pil_label)
    ax[2].set(xticks=[], yticks=[])
    ax[2].imshow(pil_pred_label)
    plt.savefig(path)
    plt.close()
