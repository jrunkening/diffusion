import torch
from neuralop.models import UNO
from diffusion.noise_scheduler import NoiseScheduler
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, activation, mode, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.normalize = nn.LayerNorm((in_c, *shape))
        self.Conv = nn.Conv2d if mode == "down" else nn.ConvTranspose2d

        self.conv0 = self.Conv(in_c, out_c, kernel_size, stride, padding)
        self.conv1 = self.Conv(out_c, out_c, kernel_size, stride, padding)
        self.conv2 = self.Conv(out_c, out_c, kernel_size, stride, padding)
        self.activation = activation

        self.shortcut = self.Conv(in_c, out_c, kernel_size, stride, padding)

    def forward(self, xs):
        xs = self.normalize(xs)

        out = self.conv0(xs)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)

        out += self.shortcut(xs)

        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("embedding_w", torch.randn(256//2))
        self.noise_scheduler = NoiseScheduler(0, 0.02, 1000)

        self.channels = [3, 16, 32, 64, 128]
        self.activation = nn.SiLU()

        self.l0 = nn.Linear(256, self.channels[0])
        self.b0 = nn.Sequential(
            ResBlock((218, 178), self.channels[0], self.channels[1], self.activation, "down"),
            ResBlock((218, 178), self.channels[1], self.channels[1], self.activation, "down"),
            ResBlock((218, 178), self.channels[1], self.channels[1], self.activation, "down"),
        )
        self.down0 = nn.Conv2d(self.channels[1], self.channels[1], kernel_size=3, stride=2, padding=1)

        self.l1 = nn.Linear(256, self.channels[1])
        self.b1 = nn.Sequential(
            ResBlock((109, 89), self.channels[1], self.channels[2], self.activation, "down"),
            ResBlock((109, 89), self.channels[2], self.channels[2], self.activation, "down"),
            ResBlock((109, 89), self.channels[2], self.channels[2], self.activation, "down"),
        )
        self.down1 = nn.Conv2d(self.channels[2], self.channels[2], kernel_size=3, stride=2, padding=1)

        self.l2 = nn.Linear(256, self.channels[2])
        self.b2 = nn.Sequential(
            ResBlock((55, 45), self.channels[2], self.channels[3], self.activation, "down"),
            ResBlock((55, 45), self.channels[3], self.channels[3], self.activation, "down"),
            ResBlock((55, 45), self.channels[3], self.channels[3], self.activation, "down"),
        )
        self.down2 = nn.Conv2d(self.channels[3], self.channels[3], kernel_size=3, stride=2, padding=1)

        self.l3 = nn.Linear(256, self.channels[3])
        self.b3 = nn.Sequential(
            ResBlock((28, 23), self.channels[3], self.channels[4], self.activation, "down"),
            ResBlock((28, 23), self.channels[4], self.channels[4], self.activation, "down"),
            ResBlock((28, 23), self.channels[4], self.channels[4], self.activation, "down"),
        )
        self.down3 = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3, stride=2, padding=1)


        self.lt3 = nn.Linear(256, self.channels[4])
        self.bt3 = nn.Sequential(
            ResBlock((14, 12), self.channels[4], self.channels[3], self.activation, "up"),
            ResBlock((14, 12), self.channels[3], self.channels[3], self.activation, "up"),
            ResBlock((14, 12), self.channels[3], self.channels[3], self.activation, "up"),
        )
        self.up3 = nn.ConvTranspose2d(self.channels[3], self.channels[3], kernel_size=(4, 3), stride=2, padding=1)

        self.lt2 = nn.Linear(256, self.channels[3])
        self.bt2 = nn.Sequential(
            ResBlock((28, 23), 2*self.channels[3], self.channels[2], self.activation, "up"),
            ResBlock((28, 23), self.channels[2], self.channels[2], self.activation, "up"),
            ResBlock((28, 23), self.channels[2], self.channels[2], self.activation, "up"),
        )
        self.up2 = nn.ConvTranspose2d(self.channels[2], self.channels[2], kernel_size=3, stride=2, padding=1)

        self.lt1 = nn.Linear(256, self.channels[2])
        self.bt1 = nn.Sequential(
            ResBlock((55, 45), 2*self.channels[2], self.channels[1], self.activation, "up"),
            ResBlock((55, 45), self.channels[1], self.channels[1], self.activation, "up"),
            ResBlock((55, 45), self.channels[1], self.channels[1], self.activation, "up"),
        )
        self.up1 = nn.ConvTranspose2d(self.channels[1], self.channels[1], kernel_size=3, stride=2, padding=1)

        self.lt0 = nn.Linear(256, self.channels[1])
        self.bt0 = nn.Sequential(
            ResBlock((109, 89), 2*self.channels[1], self.channels[0], self.activation, "up"),
            ResBlock((109, 89), self.channels[0], self.channels[0], self.activation, "up"),
            ResBlock((109, 89), self.channels[0], self.channels[0], self.activation, "up"),
        )
        self.up0 = nn.ConvTranspose2d(self.channels[0], self.channels[0], kernel_size=4, stride=2, padding=1)

    def preprocess(self, xs):
        ts = torch.randint(1, self.noise_scheduler.steps, (len(xs),))
        xs, ys = self.noise_scheduler.forward_process(xs, ts)

        return xs, ts, ys

    def embed(self, ts, linear):
        embedding = 30 * torch.outer(ts, self.embedding_w)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        embedding = self.activation(linear(embedding))
        embedding = embedding.reshape(*embedding.shape, 1, 1)

        return embedding

    def forward(self, xs, ts):
        h0 = self.down0(self.b0(xs + self.embed(ts, self.l0)))
        h1 = self.down1(self.b1(h0 + self.embed(ts, self.l1)))
        h2 = self.down2(self.b2(h1 + self.embed(ts, self.l2)))
        h3 = self.down3(self.b3(h2 + self.embed(ts, self.l3)))

        h = self.up3(self.bt3(h3 + self.embed(ts, self.lt3)))
        h = self.up2(self.bt2(torch.cat((h + self.embed(ts, self.lt2), h2), dim=1)))
        h = self.up1(self.bt1(torch.cat((h + self.embed(ts, self.lt1), h1), dim=1)))
        h = self.up0(self.bt0(torch.cat((h + self.embed(ts, self.lt0), h0), dim=1)))

        return h

    def prev(self, image, t):
        t = torch.tensor(t).reshape(1)

        alpha = self.noise_scheduler.alphas[t]
        sqrt_oneminus_alpha_bar = self.noise_scheduler.sqrt_oneminus_alpha_bar_s[t]
        sigma = self.noise_scheduler.sigmas[t]

        return (
            image - ((1 - alpha) / sqrt_oneminus_alpha_bar) * self(image, t).detach()
        ) / torch.sqrt(alpha) + sigma * torch.randn_like(image)

    def infer(self, image_dims):
        image = torch.randn(1, 3, *image_dims)
        for i in range(self.noise_scheduler.steps-1, 0, -1):
            image = self.prev(image, i)
        image = image.reshape(3, *image_dims).permute(1, 2, 0)
        image -= torch.min(image)
        image /= torch.max(image)

        return image
