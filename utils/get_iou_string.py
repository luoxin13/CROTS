import numpy as np


def get_iou_string(iou, num_classes):
    if not isinstance(iou, np.ndarray):
        iou = iou.cpu().numpy()
    ious = [str(round(iou[i] * 100, 2)) for i in range(num_classes)]
    class_16 = list(set(range(19)) - set([9, 14, 16]))
    class_13 = list(set(range(19)) - set([9, 14, 16, 4, 5, 6]))
    ious.append(str(round(np.nanmean(iou) * 100, 2)))
    ious.append(str(round(np.nanmean(iou[class_16]) * 100, 2)))
    ious.append(str(round(np.nanmean(iou[class_13]) * 100, 2)))
    return ' , '.join(ious)
