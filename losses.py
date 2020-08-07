import torch
from torch.nn import functional as F

def dice_loss(output, target, weights=None):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """
    smooth = 1
    output = output.exp()  # to convert log(softmax) output from model to softmax

    encoded_target = output.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if weights is None:
        weights = 1

    intersection = output * encoded_target
    numerator = (2 * intersection.sum(2).sum(2)) + smooth 
    denominator = output + encoded_target

    denominator = denominator.sum(2).sum(2) + smooth

    loss_per_slice = 1 - (numerator / denominator)
    loss_per_batch = (loss_per_slice.sum(0))/output.size(0)

    loss_per_channel = weights * loss_per_batch

    return loss_per_channel.sum() / output.size(1)


def dice_loss_deep_supervised(output, target, weights=None):
    """
    output :[NxCxHxW,....] Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """

    loss_total = 0
    gt1 = target[:,::8,::8]
    gt2 = target[:,::4,::4]
    gt3 = target[:,::2,::2]
    gt4 = target
    modified_target = [gt1,gt2,gt3,gt4]
    output_ = output.copy()
    for i in range(4):

        smooth = 1
        output_[i] = output_[i].exp()  # to convert log(softmax) output from model to softmax

        encoded_target = output_[i].detach() * 0
        encoded_target.scatter_(1, modified_target[i].unsqueeze(1), 1)  # one hot encoding

        if weights is None:
            weights = 1

        TP = output_[i] * encoded_target
        FP = output_[i] * (1 - encoded_target)
        FN = (1 - output_[i]) * (encoded_target)

        numerator = (TP.sum(2).sum(2)) + smooth
        denominator = (TP + 0.5*FN + 0.5*FP)

        denominator = denominator.sum(2).sum(2) + smooth
        
        loss_per_slice = 1 - (numerator / denominator)
        loss_per_batch = (loss_per_slice.sum(0))/output_[i].size(0)
            
        loss_per_channel = weights * loss_per_batch

        loss_total += loss_per_channel.sum() / output_[i].size(1)

    return loss_total

def dice_score(output, target, weights=None):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    weights : C FloatTensor
    """
    smooth = 1
    output = output.exp()

    encoded_target = output.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)  # one hot encoding

    if weights is None:
        weights = 1


    intersection = output * encoded_target
    numerator = (2 * intersection.sum(2).sum(2)) + smooth   #This for 2d slices, if for a patient intersection.sum(0).sum(1).sum(1)
    denominator = output + encoded_target

    denominator = denominator.sum(2).sum(2) + smooth

    score_per_slice = numerator / denominator
    score_per_batch = (score_per_slice.sum(0))/output.size(0)

    score_per_channel = weights * score_per_batch
    return score_per_channel
