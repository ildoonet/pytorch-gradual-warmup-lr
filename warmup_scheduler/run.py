import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.sgd import SGD

from warmup_scheduler import GradualWarmupScheduler


def plot(lr_list):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    f = plt.figure()

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    x = range(1, len(lr_list) + 1)
    plt.plot(x, lr_list)
    plt.show()


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    epochs = 20
    # scheduler_warmup is chained with lr_schduler
    lr_schduler = CosineAnnealingLR(optim, T_max=epochs - 5, eta_min=0.02)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=lr_schduler)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()
    scheduler_warmup.step()

    lr_list = list()
    for epoch in range(epochs):
        current_lr = optim.param_groups[0]['lr']

        optim.step()
        scheduler_warmup.step()

        print(epoch + 1, current_lr)
        lr_list.append(current_lr)

    plot(lr_list)
