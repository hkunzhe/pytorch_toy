import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Customize adding backdoor to transforms that speeds up preprocess.
class AddTrigger(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img):
        return add_backdoor(img, **self.kwargs)


def add_backdoor(img, **kwargs):
    trigger_loc = kwargs["trigger_loc"]
    trigger_ptn = kwargs["trigger_ptn"]
    for i, (m, n) in enumerate(trigger_loc):
        img[m, n, :] = trigger_ptn[i]  # add trigger
    return img


def plot_data(dataset, nrows=10, ncols=10):
    data_loader = DataLoader(dataset, batch_size=nrows * ncols)
    batch_data, targets, _ = next(iter(data_loader))  # fetch the first batch data
    batch_data = batch_data.permute(0, 2, 3, 1)  # convert to NHWC
    fig, axes = plt.subplots(nrows, ncols)
    fig.figsize = (12, 9)
    fig.dpi = 600
    for r in range(nrows):
        for c in range(ncols):
            idx = r * nrows + c
            axes[r][c].imshow(batch_data[idx].numpy())
            axes[r][c].set_title(str(targets[idx].item()))
            axes[r][c].axis("off")
    fig.savefig("test.png")
