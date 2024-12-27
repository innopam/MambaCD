import math
import numpy as np
import torch
import torch.nn.functional as F

def cal_class_weight(class_dist, itera, alpha=0.7):
    # 클래스 분포 출력
    formatted_dist = {label: f"{count:,}" for label, count in class_dist.items()}
#    print("클래스 분포:", formatted_dist)

    total_count = sum(class_dist.values())
    num_classes = len(class_dist)
    
    # 각 클래스의 비율 계산 및 출력
    class_percentage = {label: f"{(count / total_count * 100):.2f}%" for label, count in class_dist.items()}
    print("클래스 비율:", class_percentage)

    # 제로 디비전 체크
    counts = np.array(list(class_dist.values()))
    counts[counts == 0] = 1e-6  # 최소값으로 대체하여 제로 디비전 방지
    
    scheduler = math.log10(1000 + (itera // 100)) - 3.0

    class_weights = (total_count / (num_classes * counts)) ** (alpha - scheduler)
    norm_weights = class_weights / np.sum(class_weights)

    print("정규화된 클래스 가중치:", norm_weights)

    return norm_weights

# Focal Loss 정의
def focal_loss(preds, targets, class_weights=None, gamma=2.0, reduction="mean"):
    ce_loss = F.cross_entropy(preds, targets, reduction=reduction)
    
    batch_size, height, width = targets.shape
    class_weights = class_weights[targets]
    alpha = class_weights.view(batch_size, height, width)
    
    pt = torch.exp(-ce_loss)
    loss = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")

    return loss

# Dice Loss 정의
def dice_loss(preds, targets, smooth=0.0, eps=1e-7):
    preds = F.log_softmax(preds, dim=1).exp()
    
    bs = targets.size(0)
    num_classes = preds.size(1)
    dims = (0, 2)
    
    targets = targets.view(bs, -1)
    preds = preds.view(bs, num_classes, -1)

    targets = F.one_hot(targets, num_classes).permute(0, 2, 1)
    
    intersection = torch.sum(preds * targets, dim=dims)
    cardinality = torch.sum(preds + targets, dim=dims)
    
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    loss = 1.0 - dice_score
    
    return loss
    
def iou_loss(preds, targets, smooth=1e-6):
    preds = F.log_softmax(preds, dim=1).exp()
    bs = targets.size(0)
    num_classes = preds.size(1)
    dims = (0, 2)
    
    targets = targets.view(bs, -1)
    preds = preds.view(bs, num_classes, -1)

    targets = F.one_hot(targets, num_classes).permute(0, 2, 1)
    
    intersection = torch.sum(preds * targets, dim=dims)
    union = torch.sum(preds, dim=dims) + torch.sum(targets, dim=dims) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    loss = 1.0 - iou
    
    return loss
