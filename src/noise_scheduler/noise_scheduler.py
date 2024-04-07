
import torch

class NoiseScheduler:
    def __init__(self, diffusion_config):
        self.beta_start = diffusion_config['beta_start']
        self.beta_end = diffusion_config['beta_end']
        self.num_timesteps = diffusion_config['num_timesteps']

        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0) # cummulate product, [1,2,3,4]->[1,2,6,24]
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.sqrt_alpha_cum_prod)
    def add_noise2(self, original, noise, t): # orginal:[b, c, h, w], t:[b]
        batch_size = original.shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size) # [b]
        one_minus_sqrt_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size) #[b]


        # nsqueeze [b] to [b, 1, 1, 1]
        for _ in range(len(original.shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            one_minus_sqrt_alpha_cum_prod = one_minus_sqrt_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod.to(original.device)*original + one_minus_sqrt_alpha_cum_prod.to(original.device)*noise

    def add_noise(self, original, noise, t):  # orginal:[b, c, h, w], t:[b]
        batch_size = original.shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t]
        one_minus_sqrt_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t]

        # nsqueeze [b] to [b, 1, 1, 1]
        for _ in range(len(original.shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            one_minus_sqrt_alpha_cum_prod = one_minus_sqrt_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod.to(original.device) * original + one_minus_sqrt_alpha_cum_prod.to(
            original.device) * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        xt_device = xt.device
        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod.to(xt_device)[t]*noise_pred)/self.sqrt_alpha_cum_prod.to(xt_device)[t]
        x0 = torch.clamp(x0, -1., 1.)
        mean = xt - (self.betas.to(xt_device)[t]*noise_pred)/self.sqrt_one_minus_alpha_cum_prod.to(xt_device)[t]
        mean = mean/torch.sqrt(self.alphas.to(xt_device)[t])

        if t == 0:
            return mean, x0
        else:
            z = torch.randn_like(xt).to(xt.device)
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t-1])/(1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = torch.sqrt(variance)
            return mean + sigma*z, x0
