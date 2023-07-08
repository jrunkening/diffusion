import torch


class NoiseScheduler:
    def __init__(self, start, end, steps) -> None:
        self.start, self.end, self.steps = start, end, steps
        self.betas = torch.linspace(start, end, steps)
        self.alphas = 1 - self.betas
        self.alpha_bar_s = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar_s = torch.sqrt(self.alpha_bar_s)
        self.sqrt_oneminus_alpha_bar_s = torch.sqrt(1 - self.alpha_bar_s)
        self.sigmas = torch.sqrt(self.betas)

    def forward_process(self, xs, ts):
        noise = torch.randn_like(xs)
        sqrt_alpha_bar_s = self.sqrt_alpha_bar_s[ts]
        sqrt_oneminus_alpha_bar_s = self.sqrt_oneminus_alpha_bar_s[ts]
        sqrt_alpha_bar_s = sqrt_alpha_bar_s.reshape(*sqrt_alpha_bar_s.shape, 1, 1, 1)
        sqrt_oneminus_alpha_bar_s = sqrt_oneminus_alpha_bar_s.reshape(*sqrt_oneminus_alpha_bar_s.shape, 1, 1, 1)

        return sqrt_alpha_bar_s * xs + sqrt_oneminus_alpha_bar_s * noise, noise


NOISE_SCHEDULER = NoiseScheduler(0, 0.02, 1000)


def prev(image, t, model):
    t = t * torch.ones(1, dtype=torch.int32)

    alpha = NOISE_SCHEDULER.alphas[t]
    sqrt_oneminus_alpha_bar = NOISE_SCHEDULER.sqrt_oneminus_alpha_bar_s[t]
    sigma = NOISE_SCHEDULER.sigmas[t]
    return (
        image - \
        ((1 - alpha) / torch.sqrt(1 - sqrt_oneminus_alpha_bar)) * model(image, t)
    ) / torch.sqrt(alpha) + sigma * torch.randn(1, 3, 218, 178)

def infer(y_size, x_size):
    image = torch.randn(1, 3, y_size, x_size)
    for i in range(NOISE_SCHEDULER.steps-1, -1, -1):
        image = prev(image, 499)
    image = image.reshape(3, 218, 178).permute(1, 2, 0).detach()
    image -= torch.min(image)
    image /= torch.max(image)

    return image
