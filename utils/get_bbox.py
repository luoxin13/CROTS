import cv2
import numpy as np


def get_bbox(mask):
    """ get bbox from semantic label """
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    assert len(mask.shape) == 2
    mask_2D = mask.copy()
    obj_ids = np.unique(mask_2D)
    masks = mask_2D == obj_ids[:, None, None]
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        id_x = obj_ids[i]
        binary = masks[i].astype(np.int8)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_16U)
        for id_label in range(1, num_labels):
            temp_mask = labels == id_label
            pos = np.where(temp_mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if (xmax - xmin) > 20 and (ymax - ymin) > 20:
                boxes.append([id_x, (xmin, ymin, xmax, ymax)])
    return boxes
