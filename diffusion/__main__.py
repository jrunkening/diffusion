import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from diffusion.noise_scheduler import NOISE_SCHEDULER
from diffusion.model import Model
from neuralop.models import UNO


DATA_PATH = Path(__file__).parent.joinpath("../data")
MODEL_PATH = Path(__file__).parent.joinpath("../model")


def preprocess(xs):
    ts = torch.randint(1, NOISE_SCHEDULER.steps, (len(xs),))
    xs, ys = NOISE_SCHEDULER.forward_process(xs, ts)

    y_pos = torch.ones(xs.shape[0], 1, *xs.shape[2:])
    x_pos = torch.ones(xs.shape[0], 1, *xs.shape[2:])
    t_pos = torch.ones(xs.shape[0], 1, *xs.shape[2:])
    for i in range(xs.shape[2]):
        y_pos[:, :, i, :] *= i
    for i in range(xs.shape[3]):
        x_pos[:, :, :, i] *= i
    for i, t in enumerate(ts):
        t_pos[i, :, :, :] *= t
    xs = torch.cat((xs, y_pos), dim=1)
    xs = torch.cat((xs, x_pos), dim=1)
    xs = torch.cat((xs, t_pos), dim=1)

    return xs, ys


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (xs, _) in enumerate(dataloader):
        # add noise
        xs, ys = preprocess(xs)

        # calculate loss
        xs, ys = xs.to(device), ys.to(device)
        loss = loss_fn(model(xs), ys)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(xs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xs, _ in dataloader:
            # add noise
            xs, ys = preprocess(xs)

            # calculate loss
            xs, ys = xs.to(device), ys.to(device)
            test_loss += loss_fn(model(xs), ys).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss


def main(on_gpu=True):
    device = "cuda" if on_gpu and torch.cuda.is_available() else "cpu"

    training_data = datasets.CelebA(
        root=DATA_PATH,
        split="train",
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = datasets.CelebA(
        root=DATA_PATH,
        split="test",
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = UNO(
        in_channels=6,
        out_channels=3,
        hidden_channels=128,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        uno_out_channels=[64, 32, 32, 64],
        uno_n_modes=[[128, 128], [128, 128], [128, 128], [128, 128]],
        uno_scalings=[[1, 1], [1, 1], [1, 1], [1, 1]],
    ).to(device)
    # model = torch.load(MODEL_PATH.joinpath(os.listdir(MODEL_PATH, )[-1]))
    learning_rate = 1e-4
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 500
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
        torch.save(model.state_dict(), MODEL_PATH.joinpath("model.pth"))

    print("Done!")


if __name__ == "__main__":
    main()
