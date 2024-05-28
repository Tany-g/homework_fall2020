import numpy as np
import time
import copy
import torch
from typing_extensions import TypedDict
from typing import Tuple, List
import isaacgym
from cs285.policies.base_policy import BasePolicy
from isaacgym import gymapi
gym = gymapi.acquire_gym()
class PathDict(TypedDict):
    observation: np.ndarray
    image_obs: np.ndarray
    reward: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()["obs"][0].to("cpu").numpy()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ac = torch.tensor(ac,dtype=torch.float)
        # ac = torch.stack((ac,ac),0)
        ac = ac.expand(32,7)
        ob, rew, done, _ = env.step(ac)
        env.render()
        # add the observation after taking a step to next_obs
        ob = ob["obs"][0].to("cpu").detach().numpy()
        next_obs.append(ob)
        rewards.append(rew[0].to("cpu").detach().numpy())
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done[0]:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################


def sample_trajectory(
    env,
    policy: BasePolicy,
    max_path_length: int,
    render: bool=False,
    render_mode=('rgb_array'),
) -> PathDict:
    ob = env.reset()["obs"][0].to("cpu").numpy()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    if 'track' in env.env.model.camera_names:
                        image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obs.append(env.sim.render(height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        # 

        
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)
        
        ac = torch.tensor(ac, dtype=torch.float)
        ac = ac.expand(32, 7)
        # gym.step_graphics(env.sim)
        # gym.draw_viewer(env.viewer, env.sim, True)
        ob, rew, done, _ = env.step(ac)
        env.render()
        ob = ob["obs"][0].to("cpu").detach().numpy()
        print(ob,rew,done)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew[0].to("cpu").detach().numpy())
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done[0] or steps > max_path_length:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(
    env,
    policy: BasePolicy,
    min_timesteps_per_batch: int,
    max_path_length: int,
    render=False,
    render_mode=('rgb_array'),
) -> Tuple[List[PathDict], int]:
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.
    """
    timesteps_this_batch = 0
    paths: List[PathDict] = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path: PathDict = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += path['observation'].shape[0]

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy: BasePolicy, ntraj: int, max_path_length: int, render=False, render_mode=('rgb_array')) -> List[PathDict]:
    """
        Collect ntraj rollouts.
    """
    paths: List[PathDict] = []

    for _ in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode))

    return paths

############################################
############################################

def Path(
    obs: List[np.ndarray],
    image_obs: List[np.ndarray],
    acs: List[np.ndarray],
    rewards: List[np.ndarray],
    next_obs: List[np.ndarray], 
    terminals: List[bool],
) -> PathDict:
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
