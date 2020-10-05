import torch
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, beta=1.0, half=False):
    lam = np.random.beta(beta, beta)
    if half is False:
        batch_size = x.size()[0]
        rand_index = torch.randperm(batch_size).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), 1 - lam)
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        y = (y, y[rand_index], (torch.ones(batch_size) * lam).cuda())
    else:
        # this part can be optimized, but to be safer, I use the ugly version.
        if type(x) is tuple and len(x) == 2:
            # args.multi_grid should be true
            batch_size = x[0].size()[0]
            x_size = x[0].size()  # for for use H and W
        else:
            batch_size = x.size()[0] // 2
            x_size = x.size()
        rand_index = torch.randperm(batch_size).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(x_size, 1 - lam)
        lam = ((bbx2 - bbx1) * (bby2 - bby1) / (x_size[-1] * x_size[-2]))
        if type(x) is tuple and len(x) == 2:
            # args.multi_grid should be true
            x[0][:, :, bbx1:bbx2, bby1:bby2] = x[0][rand_index, :, bbx1:bbx2, bby1:bby2]
        else:
            x[:batch_size][:, :, bbx1:bbx2, bby1:bby2] = x[:batch_size][rand_index, :, bbx1:bbx2, bby1:bby2]
        assert len(y) == 3
        for i in range(3):
            assert y[i].shape[0] == batch_size * 2
        y[1][:batch_size] = y[0][:batch_size][rand_index]
        y[2][:batch_size] = lam

    return x, y
