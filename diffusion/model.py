import torch
from diffusion.noise_scheduler import NoiseScheduler
from torch import nn


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, ts):
        half_dim = self.dim // 2
        embedding = torch.tensor(10000).log() / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=ts.device) * -embedding)
        embedding = ts[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)

        return embedding


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
    attention_scores = nn.functional.softmax(attention_scores, dim=-1)

    out = values.bmm(attention_scores).view(n, self.out_channels, h, w) + self.l(xs)

    return out


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels, activate, mode,
            kernel_size=3, stride=1, padding=1, drop_rate=0., is_pool=False) -> None:
        super().__init__()
        self.groups = min(in_channels, out_channels)
        self.Conv = nn.Conv2d if mode == "down" else nn.ConvTranspose2d
        if is_pool:
            self.pool = nn.AvgPool2d(2) if mode == "down" else nn.ConvTranspose2d(out_channels, out_channels, 2, 2, 0)
        else:
            self.pool = nn.Identity()
        self.embed = SinusoidalStepEmbedding(256)
        self.l = nn.Linear(256, out_channels)

        self.normalize_in = nn.GroupNorm(self.groups, in_channels)
        self.conv0 = self.Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.normalize_out = nn.GroupNorm(self.groups, out_channels)
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.conv1 = self.Conv(out_channels, out_channels, kernel_size, stride, padding)
        self.activate = activate

        self.shortcut = self.Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, xs, ts):
        out = self.activate(self.normalize_in(xs))
        out = self.conv0(out)
        out += self.activate(self.l(self.embed(ts))).view(*out.shape[0:2], 1, 1)
        out = self.dropout(self.activate(self.normalize_out(out)))
        out = self.conv1(out)
        out += self.shortcut(xs)

        return self.pool(out)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_scheduler = NoiseScheduler(0, 0.02, 1000)

        self.channels = [4, 16, 32, 64, 128, 256, 512]
        self.activate = nn.SiLU()

        self.lift = nn.Linear(3, self.channels[0])

        self.down0 = ResBlock(self.channels[0], self.channels[1], self.activate, "down")
        self.down1 = ResBlock(self.channels[1], self.channels[2], self.activate, "down")
        self.down2 = ResBlock(self.channels[2], self.channels[3], self.activate, "down")
        self.down3 = ResBlock(self.channels[3], self.channels[4], self.activate, "down")

        self.attention = SelfAttention(self.channels[4], self.channels[4])

        self.up3 = ResBlock(2*self.channels[4], self.channels[3], self.activate, "up")
        self.up2 = ResBlock(2*self.channels[3], self.channels[2], self.activate, "up")
        self.up1 = ResBlock(2*self.channels[2], self.channels[1], self.activate, "up")
        self.up0 = ResBlock(2*self.channels[1], self.channels[0], self.activate, "up")

        self.proj = nn.Linear(self.channels[0], 3)

    def preprocess(self, xs):
        ts = torch.randint(1, self.noise_scheduler.steps, (len(xs),))
        xs, ys = self.noise_scheduler.forward_process(xs, ts)

        return xs, ts, ys

    def forward(self, xs, ts):
        xs = self.lift(xs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        h0 = self.down0(xs, ts)
        h1 = self.down1(h0, ts)
        h2 = self.down2(h1, ts)
        h3 = self.down3(h2, ts)

        h = self.attention(h3)

        h = self.up3(torch.cat((h, h3), dim=1), ts)
        h = self.up2(torch.cat((h, h2), dim=1), ts)
        h = self.up1(torch.cat((h, h1), dim=1), ts)
        h = self.up0(torch.cat((h, h0), dim=1), ts)

        h = self.proj(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return h

    def prev(self, image, t):
        device = image.device

        t = torch.tensor(t, device=device).reshape(1)
        alpha = self.noise_scheduler.alphas.to(device)[t]
        sqrt_oneminus_alpha_bar = self.noise_scheduler.sqrt_oneminus_alpha_bar_s.to(device)[t]
        sigma = self.noise_scheduler.sigmas.to(device)[t]

        noise_scale = ((1 - alpha) / sqrt_oneminus_alpha_bar)
        noise = self(image, t).detach()
        normalize_scale = torch.sqrt(alpha)
        perturb = sigma * torch.randn_like(image)

        return (image - noise_scale * noise) / normalize_scale + perturb

    def infer(self, ch, h, w):
        image = torch.randn(1, ch, h, w, device=next(self.parameters()).device)
        for i in range(self.noise_scheduler.steps-1, 0, -1):
            image = self.prev(image, i)
        image = image.reshape(ch, h, w).permute(1, 2, 0)
        image -= torch.min(image)
        image /= torch.max(image)

        return image.to("cpu")
