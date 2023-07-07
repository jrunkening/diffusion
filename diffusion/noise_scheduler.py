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
