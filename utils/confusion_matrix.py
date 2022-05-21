def get_iou(cm_value, drop_last=True):
    if drop_last:
        cm_value = cm_value[:-1, :-1]
    inter = cm_value.diag()
    union = cm_value.sum(dim=0) + cm_value.sum(dim=1) - inter
    iou_classes = inter / union
    return iou_classes


def get_acc(cm_value, drop_last=True):
    if drop_last:
        cm_value = cm_value[:-1, :-1]
    inter = cm_value.diag()
    preds = cm_value.sum(dim=0)
    acc_classes = inter / preds
    return acc_classes


def get_acc_and_iou(cm_value, drop_last=True):
    if drop_last:
        cm_value = cm_value[:-1, :-1]
    inter = cm_value.diag()
    preds = cm_value.sum(dim=0)
    targets = cm_value.sum(dim=1)
    acc_classes = inter / preds
    iou_classes = inter / (preds + targets - inter)
    return acc_classes, iou_classes
