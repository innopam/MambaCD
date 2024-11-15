import torch
import torch.nn.functional as F

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
