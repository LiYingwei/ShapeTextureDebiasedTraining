from __future__ import print_function, absolute_import
import torch
__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_and_perclass(output, target, topk=(1,), numclasses=200):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    # top1 per-class stats
    num = target.bincount()
    wt = (pred[0] == target).int()
    correct = target.bincount(wt)
    num_per_class = torch.zeros(numclasses).int()
    correct_per_class = torch.zeros(numclasses).int()
    num_per_class[:len(num)] = num
    correct_per_class[:len(correct)] = correct    

    return *res, num_per_class, correct_per_class