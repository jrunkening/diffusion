from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from diffusion.noise_scheduler import NoiseScheduler
from diffusion.model import Model


DATA_PATH = Path(__file__).parent.joinpath("../data")


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (xs, _) in enumerate(dataloader):
        # add noise
        ts = torch.randint(1, model.noise_scheduler.steps, (len(xs),))
        xs, ys = model.noise_scheduler.forward_process(xs, ts)

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


def test_loop(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for xs, _ in dataloader:
            # add noise
            ts = torch.randint(1, model.noise_scheduler.steps, (len(xs),))
            xs, ys = model.noise_scheduler.forward_process(xs, ts)

            # calculate loss
            xs, ts, ys = xs.to(device), ts.to(device), ys.to(device)
            test_loss += loss_fn(model(xs, ts), ys).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


def main(on_gpu=True):
    device = "cuda" if on_gpu and torch.cuda.is_available() else "cpu"

    training_data = datasets.CelebA(
        root=DATA_PATH,
        split="train",
        # target_type=[],
        transform=transforms.ToTensor(),
        download=True
    )
    test_data = datasets.CelebA(
        root=DATA_PATH,
        split="test",
        # target_type=[],
        transform=transforms.ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512, shuffle=True)

    model = Model(NoiseScheduler(0, 0.02, 500)).to(device)
    learning_rate = 1e-3
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)
    print("Done!")


if __name__ == "__main__":
    main()
