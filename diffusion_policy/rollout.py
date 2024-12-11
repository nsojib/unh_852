import collections
import torch 
import numpy as np
from tqdm import tqdm

from pusht_data_utils import get_data_stats, normalize_data, unnormalize_data, PushTImageDatasetFromHDF5

def rollout(env, nets, ddpm, obs_horizon, action_horizon, stats, sample_shape, device, seed, max_steps=200):
        
    env.seed(200+seed)
    obs = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0
    success=False
    with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
        while not done:
            B = 1 
            images = np.stack([x['image'] for x in obs_deque])
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
 
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos']) 
            nimages = images
 
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32) 
            nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
     
 
            with torch.no_grad(): 
                image_features = nets['vision_encoder'](nimages) 
                obs_features = torch.cat([image_features, nagent_poses], dim=-1) 
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                naction,xts=ddpm.sample_ddpm(1, sample_shape, obs_cond)
                

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:] # (action_horizon, action_dim)

            # receding horizon control: execute n action before re-planning
            for i in range(len(action)):
                obs, reward, done, info = env.step(action[i])
                obs_deque.append(obs)
                
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    success=True
                    break

    return max(rewards) , success, imgs