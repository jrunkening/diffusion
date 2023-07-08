import torch
from neuralop.models import UNO
from diffusion.noise_scheduler import NoiseScheduler


class Model:
    def __init__(self) -> None:
        self.noise_scheduler = NoiseScheduler(0, 0.02, 1000)
        self.operator = UNO(
            in_channels=6,
            out_channels=3,
            hidden_channels=128,
            lifting_channels=256,
            projection_channels=256,
            n_layers=4,
            uno_out_channels=[64, 32, 32, 64],
            uno_n_modes=[[128, 128], [128, 128], [128, 128], [128, 128]],
            uno_scalings=[[1, 1], [1, 1], [1, 1], [1, 1]],
        )

    def cat_pos_info(self, xs, ts):
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

        return xs

    def preprocess(self, xs):
        ts = torch.randint(1, self.noise_scheduler.steps, (len(xs),))
        xs, ys = self.noise_scheduler.forward_process(xs, ts)

        return self.cat_pos_info(xs, ts), ys

    def prev(self, image, t):
        t = torch.tensor(t).reshape(1)

        alpha = self.noise_scheduler.alphas[t]
        sqrt_oneminus_alpha_bar = self.noise_scheduler.sqrt_oneminus_alpha_bar_s[t]
        sigma = self.noise_scheduler.sigmas[t]

        return (
            image - ((1 - alpha) / sqrt_oneminus_alpha_bar) * self.operator(self.cat_pos_info(image, t)).detach()
        ) / torch.sqrt(alpha) + sigma * torch.randn_like(image)

    def infer(self, image_dims):
        image = torch.randn(1, 3, *image_dims)
        for i in range(self.noise_scheduler.steps-1, -1, -1):
            image = self.prev(image, i)
        image = image.reshape(3, *image_dims).permute(1, 2, 0)
        # image -= torch.min(image)
        # image /= torch.max(image)

        return image
