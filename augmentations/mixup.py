import torch
import numpy as np


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (1. - lam) * criterion(pred, y_a) + lam * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, half=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)

    if half is False:
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        x = (1 - lam) * x + lam * x[index, :]  # note: here we should be consistent with the mixup_criterion
        y = (y, y[index], (torch.ones(batch_size) * lam).cuda())
    else:
        if type(x) is tuple and len(x) == 2:
            # args.multi_grid should be true
            batch_size = x[0].size()[0]
        else:
            batch_size = x.size()[0] // 2
        index = torch.randperm(batch_size).cuda()
        if type(x) is tuple and len(x) == 2:
            # args.multi_grid should be true
            x[0][:] = (1 - lam) * x[0][:] + lam * x[0][index, :]
        else:
            x[:batch_size] = (1 - lam) * x[:batch_size] + lam * x[:batch_size][index, :]
        assert len(y) == 3
        for i in range(3):
            assert y[i].shape[0] == batch_size * 2
        y[1][:batch_size] = y[0][:batch_size][index]
        y[2][:batch_size] = lam

    return x, y
