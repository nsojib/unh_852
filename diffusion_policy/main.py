import numpy as np
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F


from pusht_data_utils import PushTImageDatasetFromHDF5
from vision_model import ResidualBlock, ResNetFe, replace_bn_with_gn
from noise_predictor_model import ConditionalUnet1D
from myddpm import MyScheduler, MyDDPM
from rollout import rollout
import argparse 

import sys  
# sys.path.append('/home/carl_lab/diffusion_policy/')  
sys.path.append('/home/ns1254/diffusion_policy/') 

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
 
 
def main(args): 
    
    # hyperparameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    vision_feature_dim = 512
    lowdim_obs_dim = 2
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device 

    hdf5_file_name = args.hdf5_file
    hdf5_filter_key = args.hdf5_filter_key
    num_epochs = args.epochs
    seed=args.seed
    eval_epochs=args.eval_epochs
    pos_encoder=args.pos_encoder
    
    # loading the dataset
    dataset = PushTImageDatasetFromHDF5(
        hdf5_file_name=hdf5_file_name,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        hdf5_filter_key=hdf5_filter_key
    )
    stats=dataset.stats

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # neural networks (vision encoder and noise-predictor unet)
    vision_encoder = ResNetFe(ResidualBlock, [2, 2]) 
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        pos_encoder_model=pos_encoder
    )

    #print number of parameters for each network
    print('Number of parameters for vision encoder:', sum(p.numel() for p in vision_encoder.parameters()))
    print('Number of parameters for noise predictor:', sum(p.numel() for p in noise_pred_net.parameters()))



    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })
    _ = nets.to(device)


    # diffusion model (scheduler and sampler)
    num_diffusion_iters = 100
    sample_shape=(pred_horizon, action_dim) 

    noise_scheduler=MyScheduler(T=num_diffusion_iters, device=device)
    ddpm=MyDDPM(noise_scheduler, nets['noise_pred_net'], device=device)
 

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6) 

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    # main training loop
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        
        for epoch_idx in tglobal:
                
            epoch_loss = list()
            for nbatch in dataloader:
                
                #preprocessing the input and target data
                nimage = nbatch['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]
    
                image_features = nets['vision_encoder']( nimage.flatten(end_dim=1))           #extracting image features
                image_features = image_features.reshape(*nimage.shape[:2],-1) 
    
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)                #concatenating input modes (image feature, agent pos)
                obs_cond = obs_features.flatten(start_dim=1) 
    
                timesteps = torch.randint( 0, noise_scheduler.T, (B,), device=device ).long()
                noisy_actions , eps= noise_scheduler.get_xt(naction, timesteps)               #calculating noise for the batch

                eps_theta = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)    #predicting noise for the batch

                loss = nn.functional.mse_loss(eps_theta, eps)


                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                lr_scheduler.step()


                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
            
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    print("Training done") 
    
    
    # evaluating trained policy on the environment
    env = PushTImageEnv()

    nets.eval()
    pass 
    rewards=[]
    success=[]
    lengths=[] 

    np.random.seed(seed)
    torch.manual_seed(seed)

    for i in range(eval_epochs):
        reward, suc, imgs =  rollout(env, nets, ddpm, obs_horizon, action_horizon, stats, sample_shape, device, seed, max_steps=200)
        rewards.append(reward)
        success.append(suc)
        lengths.append(len(imgs))

    print('Mean Reward: ', np.mean(rewards))
    print('Mean Length: ', np.mean(lengths))

    mean_r= np.mean(rewards)
    #save the model
    torch.save(nets, f'trained_model_{mean_r}.pth')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--hdf5_file', type=str, required=True)
    parser.add_argument('--hdf5_filter_key', type=str, default=None) 
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--eval_epochs', type=int, default=50)
    parser.add_argument('--pos_encoder', type=str, default='custom')  #[custom, default]
    args = parser.parse_args()
    main(args)
    

# python main.py --hdf5_file ../data/pusht/pusht_v7_zarr_206.hdf5  --pos_encoder custom
# python main.py --hdf5_file ../data/pusht/pusht_v7_zarr_206.hdf5  --pos_encoder default



# Number of parameters for vision encoder: 749120
# Number of parameters for noise predictor: 43189762