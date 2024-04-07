import numpy as np
import time

############################################
############################################


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob = env.reset()  # HINT: should be the output of resetting the env [OK]

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:  # rgb_array is the mode used in the monitor wrapper, which is used to record videos
                if hasattr(env, 'sim'):  # hasattr is used to check if the object has the attribute 'sim'
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    # env.sim.render is used to render the environment
                    # camera_name is the name of the camera used to render the environment
                    # height and width are the height and width of the image
                    # [::-1] is used to reverse the order of the elements in the array, which is used to reverse the image, because the image is stored in the array in the reverse order???
                else:
                    # if the environment does not have the attribute 'sim', then we use the render function of the environment
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                # if the mode is human, then we use the render function of the environment
                # human mode is used to render the environment in a window
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)  # we sleep for the time step of the environment model

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)  # HINT: query the policy's get_action function [OK]
        ac = ac[0]  # the action is the first element of the array, because the get_action function returns a tuple, where the first element is the action
        # the action is the first element of the tuple, because the get_action function returns a tuple, where the first element is the action
        # the second element of the tuple is the dictionary, which contains the information about the action???
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = int(done or steps >= max_path_length)  # HINT: this is either 0 or 1
        terminals.append(rollout_done)  # we append the value of rollout_done to the terminals array

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode))  # here we append the path to the paths array
        timesteps_this_batch += get_pathlength(paths[-1])  # here we add the length of the path to the timesteps_this_batch, using the get_pathlength function
        # using [-1] we get the last element of the array, which is the path that we just appended to the array
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into the sampled_paths list.
    """
    sampled_paths = []

    for _ in range(ntraj):
        sampled_paths.append(sample_trajectory(env, policy, max_path_length, render, render_mode))

    return sampled_paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
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


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
