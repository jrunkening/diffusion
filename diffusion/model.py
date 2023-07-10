import torch
from neuralop.models import UNO
from diffusion.noise_scheduler import NoiseScheduler
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("embedding_w", torch.randn(256//2))
        self.noise_scheduler = NoiseScheduler(0, 0.02, 1000)

        self.channels = [3, 16, 32, 64, 128]
        self.act = nn.GELU()

        self.conv1 = nn.Conv2d(self.channels[0], self.channels[1], (3, 3), padding=(1, 1))
        self.d1 = nn.Linear(256, self.channels[1])
        self.pool1 = nn.MaxPool2d((2, 2))
        self.n1 = nn.GroupNorm(16, self.channels[1])
        self.conv2 = nn.Conv2d(self.channels[1], self.channels[2], (3, 3), padding=(1, 1))
        self.d2 = nn.Linear(256, self.channels[2])
        self.pool2 = nn.MaxPool2d((2, 2), stride=(1, 1))
        self.n2 = nn.GroupNorm(32, self.channels[2])
        self.conv3 = nn.Conv2d(self.channels[2], self.channels[3], (3, 3), padding=(1, 1))
        self.d3 = nn.Linear(256, self.channels[3])
        self.pool3 = nn.MaxPool2d((2, 2))
        self.n3 = nn.GroupNorm(32, self.channels[3])
        self.conv4 = nn.Conv2d(self.channels[3], self.channels[4], (3, 3), padding=(1, 1))
        self.d4 = nn.Linear(256, self.channels[4])
        self.pool4 = nn.MaxPool2d((2, 2))
        self.n4 = nn.GroupNorm(32, self.channels[4])

        self.up4 = nn.ConvTranspose2d(self.channels[4], self.channels[3], (2, 2), stride=(2, 2))
        self.dt4 = nn.Linear(256, self.channels[3])
        self.convt4 = nn.ConvTranspose2d(self.channels[3], self.channels[3], (3, 3), padding=(1, 1))
        self.nt4 = nn.GroupNorm(32, self.channels[3])
        self.up3 = nn.ConvTranspose2d(2*self.channels[3], self.channels[2], (2, 2), stride=(2, 2))
        self.dt3 = nn.Linear(256, self.channels[2])
        self.convt3 = nn.ConvTranspose2d(self.channels[2], self.channels[2], (3, 3), padding=(1, 1))
        self.nt3 = nn.GroupNorm(32, self.channels[2])
        self.up2 = nn.ConvTranspose2d(2*self.channels[2], self.channels[1], (2, 2), stride=(1, 1))
        self.dt2 = nn.Linear(256, self.channels[1])
        self.convt2 = nn.ConvTranspose2d(self.channels[1], self.channels[1], (3, 3), padding=(1, 1))
        self.nt2 = nn.GroupNorm(16, self.channels[1])
        self.up1 = nn.ConvTranspose2d(2*self.channels[1], self.channels[0], (2, 2), stride=(2, 2))
        self.dt1 = nn.Linear(256, self.channels[0])
        self.convt1 = nn.ConvTranspose2d(self.channels[0], self.channels[0], (3, 3), padding=(1, 1))
        self.nt1 = nn.GroupNorm(3, self.channels[0])

    def preprocess(self, xs):
        ts = torch.randint(1, self.noise_scheduler.steps, (len(xs),))
        xs, ys = self.noise_scheduler.forward_process(xs, ts)

        return xs, ts, ys

    def embed(self, ts, linear):
        embedding = 30 * torch.outer(ts, self.embedding_w)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        embedding = self.act(linear(embedding))
        embedding = embedding.reshape(*embedding.shape, 1, 1)

        return embedding

    def forward(self, xs, ts):
        h1 = self.conv1(xs)
        h1 += self.embed(ts, self.d1)
        h1 = self.pool1(h1)
        h1 = self.n1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.embed(ts, self.d2)
        h2 = self.pool2(h2)
        h2 = self.n2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.embed(ts, self.d3)
        h3 = self.pool3(h3)
        h3 = self.n3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.embed(ts, self.d4)
        h4 = self.pool4(h4)
        h4 = self.n4(h4)
        h4 = self.act(h4)


        h = self.up4(h4)
        h += self.embed(ts, self.dt4)
        h = self.convt4(h)
        h = self.nt4(h)
        h = self.act(h)

        h = self.up3(torch.cat((h, h3), dim=1))
        h += self.embed(ts, self.dt3)
        h = self.convt3(h)
        h = self.nt3(h)
        h = self.act(h)

        h = self.up2(torch.cat((h, h2), dim=1))
        h += self.embed(ts, self.dt2)
        h = self.convt2(h)
        h = self.nt2(h)
        h = self.act(h)

        h = self.up1(torch.cat((h, h1), dim=1))
        h = self.convt1(h)
        h = self.nt1(h)

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
