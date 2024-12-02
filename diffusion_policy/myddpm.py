import torch

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class MyScheduler: 
    def __init__(self, T=100, scheduler_type="linear", beta_start=1e-4, beta_end=0.02, device='cuda'):
        # self.betas=torch.linspace(beta_start, beta_end, T).to(device)
        if scheduler_type=="linear": 
            betas=torch.linspace(beta_start, beta_end, T).to(device)
        elif scheduler_type=="cosine":
            betas=cosine_beta_schedule(T)
        else:
            raise NotImplementedError(f"Scheduler type {scheduler_type} not implemented")
        
        self.betas=betas
        self.T = T
        self.alphas=1.0 - self.betas
        self.device=device

        self.sqrt_one_minus_betas = torch.sqrt(1.0 - self.betas)
        
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) 
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.sqrt_betas = torch.sqrt(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)

    def extract(self, a, t, x_shape): 
        """
        collect the values at time t from a and reshape it to x_shape
        
        a: precomputed one dimensional values
        t: time steps (scalar or tensor)
        x_shape: shape of the tensor to be returned 
        """
        
        b=t.shape[0]
        out=a.gather(-1, t)
        rt=out.reshape(b, *((1,) * (len(x_shape) - 1)))  #batch, unpack([1]*rest of the dimension)
        return rt  

    def get_xt(self, x0, t):
        """
        compute noisy image xt from x0 at time t 
        """ 
        eps=torch.randn_like(x0)
        # xt=x0 * self.sqrt_alpha_bars[t] + self.sqrt_one_minus_alpha_bars[t] * eps
        xt = x0 * self.extract(self.sqrt_alpha_bars, t, x0.shape) + eps * self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return xt, eps


class MyDDPM:
    def __init__(self, scheduler, noise_predictor_net, device='cuda'): 
        self.scheduler=scheduler
        self.noise_predictor_net=noise_predictor_net
        self.device=device
    
    def forward(self, x0):
        """ 
        noise x0 at random time t
        return noise, predicted noise
        """
        B= x0.shape[0]
        t=torch.randint(low=0, high=self.scheduler.T, size=(B,)).long().to(self.device)
        xt, eps=self.scheduler.get_xt(x0, t)
        eps_pred=self.noise_predictor_net(xt, t)
        return xt, eps, eps_pred
    
    
    def x_t_minus_1_from_x_t(self, t, x_t, eps_theta): 
        """ 
        Algorithm 2 in the DDPM paper
        """
        sqrt_alpha=self.scheduler.extract(self.scheduler.sqrt_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_bar=self.scheduler.extract(self.scheduler.sqrt_one_minus_alpha_bars, t, x_t.shape)
        beta=self.scheduler.extract(self.scheduler.betas, t, x_t.shape)
        x_t_minus_1 = (1 / sqrt_alpha) * (x_t - ( beta / sqrt_one_minus_alpha_bar ) * eps_theta)
        
        # x_t_minus_1 = (1 / self.scheduler.sqrt_alphas[t]) * (x_t - ( self.scheduler.betas[t] / self.scheduler.sqrt_one_minus_alpha_bars[t] ) * eps_theta)
        return x_t_minus_1
    
    def sample_ddpm(self, nsamples, sample_shape, obs_cond):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = torch.randn(size=(nsamples, *sample_shape)).to(self.device)   #start from random noise
            xts = [x]
            for it in range(self.scheduler.T-1, 0, -1):
                t=torch.tensor([it]).repeat_interleave(nsamples, dim=0).long().to(self.device)
                # eps_theta = self.noise_predictor_net(x, t) 
                
                eps_theta = self.noise_predictor_net(
                    sample=x,
                    timestep=t,
                    global_cond=obs_cond
                ) 
                
                # See DDPM paper between equations 11 and 12
                x = self.x_t_minus_1_from_x_t(t, x, eps_theta) 
                if it > 1: # No noise for t=0
                    z = torch.randn(size=(nsamples, *sample_shape)).to(self.device)  
                    sqrt_beta=self.scheduler.extract(self.scheduler.sqrt_betas, t, x.shape)       #use fixed varience.
                    x += sqrt_beta* z
                xts += [x]
            return x, xts
        

