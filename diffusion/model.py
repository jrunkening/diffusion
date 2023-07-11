import torch
from diffusion.noise_scheduler import NoiseScheduler
from torch import nn


class SelfAttention(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.calc_querys = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride, padding, bias=False)
    self.calc_keys = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride, padding, bias=False)
    self.calc_values = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride, padding, bias=False)

    self.l = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)

  def forward(self, xs):
    n, _, h, w = xs.shape

    querys = self.calc_querys(xs).view(n, self.out_channels, -1)
    keys = self.calc_keys(xs).view(n, self.out_channels, -1).transpose(2, 1)
    values = self.calc_values(xs).view(n, self.out_channels, -1)

    attention_scores = keys.bmm(querys)
    assert attention_scores.shape == (n, h*w, h*w)
    attention_scores = nn.functional.softmax(attention_scores, dim=-1)

    out = values.bmm(attention_scores).view(n, self.out_channels, h, w) + self.l(xs)

    return out


class ResBlock(nn.Module):
    def __init__(
            self,
            groups, in_channels, out_channels, activate, mode,
            kernel_size=3, stride=1, padding=1, drop_rate=0.) -> None:
        super().__init__()
        self.Conv = nn.Conv2d if mode == "down" else nn.ConvTranspose2d

        self.normalize_in = nn.GroupNorm(groups, in_channels)
        self.conv0 = self.Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.normalize_out = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.conv1 = self.Conv(out_channels, out_channels, kernel_size, stride, padding)
        self.activate = activate

        self.shortcut = self.Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, xs, ts_embadding):
        out = self.activate(self.normalize_in(xs))
        out = self.conv0(out)
        out += ts_embadding
        out = self.dropout(self.activate(self.normalize_out(out)))
        out = self.conv1(out)

        return out + self.shortcut(xs)


class DownSampling(nn.Module):
    def __init__(self, group, in_channels, out_channels, activate, Pool, embedding_w) -> None:
        super().__init__()
        self.activate = activate
        self.register_buffer("embedding_w", embedding_w)
        self.ts_embadding_dim = self.embedding_w.size(0) * 2

        self.l = nn.Linear(self.ts_embadding_dim, out_channels)
        self.res = ResBlock(group, in_channels, out_channels, activate, "down")
        self.a = SelfAttention(out_channels, out_channels)
        self.down = Pool(2)

    def embed(self, ts):
        embedding = 30 * torch.outer(ts, self.embedding_w)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        embedding = (self.l(embedding))
        embedding = embedding.reshape(*embedding.shape, 1, 1)

        return embedding

    def forward(self, xs, ts):
        return self.down(self.a(self.res(xs, self.embed(ts))))


class UpSampling(nn.Module):
    def __init__(self, group, in_channels, out_channels, activate, embedding_w) -> None:
        super().__init__()
        self.activate = activate
        self.register_buffer("embedding_w", embedding_w)
        self.ts_embadding_dim = self.embedding_w.size(0) * 2

        self.l = nn.Linear(self.ts_embadding_dim, out_channels)
        self.res = ResBlock(group, in_channels, out_channels, activate, "up")
        self.a = SelfAttention(out_channels, out_channels)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def embed(self, ts):
        embedding = 30 * torch.outer(ts, self.embedding_w)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        embedding = (self.l(embedding))
        embedding = embedding.reshape(*embedding.shape, 1, 1)

        return embedding

    def forward(self, xs, ts):
        return self.up(self.a(self.res(xs, self.embed(ts))))


class Mid(nn.Module):
    def __init__(self, group, in_channels, out_channels, activate, embedding_w) -> None:
        super().__init__()
        self.activate = activate
        self.register_buffer("embedding_w", embedding_w)
        self.ts_embadding_dim = self.embedding_w.size(0) * 2

        self.l = nn.Linear(self.ts_embadding_dim, out_channels)
        self.res0 = ResBlock(group, in_channels, out_channels, activate, "down")
        self.a = SelfAttention(out_channels, out_channels)
        self.res1 = ResBlock(group, out_channels, out_channels, activate, "down")

    def embed(self, ts):
        embedding = 30 * torch.outer(ts, self.embedding_w)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        embedding = (self.l(embedding))
        embedding = embedding.reshape(*embedding.shape, 1, 1)

        return embedding

    def forward(self, xs, ts):
        embedding = self.embed(ts)
        return self.res1(self.a(self.res0(xs, embedding)), embedding)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("embedding_w", torch.randn(256//2))
        self.noise_scheduler = NoiseScheduler(0, 0.01, 1000)

        self.channels = [4, 16, 32, 64, 128, 256, 512]
        self.activate = nn.SiLU()
        self.Pool = nn.AvgPool2d

        self.lift = nn.Conv2d(3, self.channels[0], 3, 1, 1)
        self.down0 = DownSampling(self.channels[0], self.channels[0], self.channels[1], self.activate, self.Pool, self.embedding_w)
        self.down1 = DownSampling(self.channels[1], self.channels[1], self.channels[2], self.activate, self.Pool, self.embedding_w)
        self.down2 = DownSampling(self.channels[2], self.channels[2], self.channels[3], self.activate, self.Pool, self.embedding_w)
        self.down3 = DownSampling(self.channels[2], self.channels[3], self.channels[4], self.activate, self.Pool, self.embedding_w)
        self.mid0 = Mid(self.channels[2], self.channels[4], self.channels[5], self.activate, self.embedding_w)
        self.mid1 = Mid(self.channels[2], self.channels[5], self.channels[4], self.activate, self.embedding_w)
        self.up3 = UpSampling(self.channels[2], 2*self.channels[4], self.channels[3], self.activate, self.embedding_w)
        self.up2 = UpSampling(self.channels[2], 2*self.channels[3], self.channels[2], self.activate, self.embedding_w)
        self.up1 = UpSampling(self.channels[1], 2*self.channels[2], self.channels[1], self.activate, self.embedding_w)
        self.up0 = UpSampling(self.channels[0], 2*self.channels[1], self.channels[0], self.activate, self.embedding_w)
        self.proj = nn.ConvTranspose2d(self.channels[0], 3, 3, 1, 1)

    def preprocess(self, xs):
        ts = torch.randint(1, self.noise_scheduler.steps, (len(xs),))
        xs, ys = self.noise_scheduler.forward_process(xs, ts)

        return xs, ts, ys

    def forward(self, xs, ts):
        xs = self.lift(xs)

        h0 = self.down0(xs, ts)
        h1 = self.down1(h0, ts)
        h2 = self.down2(h1, ts)
        h3 = self.down3(h2, ts)

        h = self.mid0(h3, ts)
        h = self.mid1(h, ts)

        h = self.up3(torch.cat((h, h3), dim=1), ts)
        h = self.up2(torch.cat((h, h2), dim=1), ts)
        h = self.up1(torch.cat((h, h1), dim=1), ts)
        h = self.up0(torch.cat((h, h0), dim=1), ts)

        h = self.proj(h)

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
