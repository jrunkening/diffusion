import os
from pathlib import Path
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusion.model import Model


DATA_PATH = Path(__file__).parent.joinpath("../data")
MODEL_PATH = Path(__file__).parent.joinpath("../model")


def train_loop(dataloader, model: Model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (xs, _) in enumerate(dataloader):
        # add noise
        xs, ts, ys = model.preprocess(xs)

        # calculate loss
        xs, ts, ys = xs.to(device), ts.to(device), ys.to(device)
        loss = loss_fn(model(xs, ts), ys)

        # back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(xs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model: Model, loss_fn, device):
    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xs, _ in dataloader:
            # add noise
            xs, ts, ys = model.preprocess(xs)

            # calculate loss
            xs, ts, ys = xs.to(device), ts.to(device), ys.to(device)
            test_loss += loss_fn(model(xs, ts).detach(), ys).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss


def main(on_gpu=True):
    device = "cuda" if on_gpu and torch.cuda.is_available() else "cpu"

    training_data = datasets.CIFAR10(
        root=DATA_PATH,
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = datasets.CIFAR10(
        root=DATA_PATH,
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH.joinpath(os.listdir(MODEL_PATH, )[-1])))
    learning_rate = 5e-6
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 500

    for t in range(epochs):
        model = model.to(device)

        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

        model = model.to("cpu")
        torch.save(model.state_dict(), MODEL_PATH.joinpath("model.pth"))

    print("Done!")


if __name__ == "__main__":
    main()
