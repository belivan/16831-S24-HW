from collections import OrderedDict
import numpy as np
import time

import gym
import torch

from rob831.infrastructure import pytorch_util as ptu
from rob831.infrastructure.logger import Logger
from rob831.infrastructure import utils
import pickle 

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
# MAX_VIDEO_LEN = 40  # we overwrite this in the code below


def make_env(env_name):
    if env_name == 'Ant-v2':
        return gym.make(env_name)
    else:
        return gym.make(env_name)


class RL_Trainer(object):  # rl = reinforcement learning

    def __init__(self, params):  # we get params from the main file, which is a dictionary

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session, TF = tensorflow
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds, meaning that the random numbers generated will be the same every time the code is run
        seed = self.params['seed']  # seed is a random number
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = make_env(self.params['env_name'])
        self.env.reset(seed=seed)

        # Maximum length for episodes, meaning the maximum number of steps in an episode
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        # self.MAX_VIDEO_LEN = self.params['ep_len']
        self.MAX_VIDEO_LEN = 40  # max number of steps in an episode

        # Is this env continuous, or self.discrete? we check the action space with the gym library
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete
        # might have to make things discrete if we want to use a discrete policy

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        # observation_space.shape[0] is the number of dimensions of the observation space
        # action space is a list of possible actions, observation space is a list of possible observations
        # action_space.n is the number of possible actions, action_space.shape[0] is the number of dimensions of the action space
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):  # dir returns a list of all the attributes in the object
            self.fps = 1 / self.env.model.opt.timestep
        else:
            self.fps = self.env.metadata['render_fps']

        #############
        ## AGENT
        #############
        # agent is the thing that interacts with the environment
        # expert policy is the policy that we want to learn from
        # student policy is the policy that we are learning, the agent is the student policy
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])
        # agent params can be the learning rate, the number of layers in the neural network, etc.

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy: the current policy using which we collect data
        :param eval_policy: the policy we use to evaluate the current policy
        :param initial_expertdata: path to expert data pkl file
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy: the policy we are trying to imitate (if doing dagger), otherwise not used. instead we use the expert policy to relabel the data
        """

        # init vars at beginning of training
        self.total_envsteps = 0 # total number of steps in the environment, which is the sum of the number of steps in each episode
        self.start_time = time.time()

        for itr in range(n_iter): # n_iter is the number of iterations we want to run the training loop for
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
                # if we want to log videos, we set log_video to True
            else:
                self.log_video = False

            # decide if metrics should be logged, which is the case if we are at the iteration where we want to log metrics
            # metrics are things like the average return, the standard deviation of the return, the maximum return, the minimum return, the average episode length, etc.
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy,
                self.params['batch_size']
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            # training_returns is a tuple, where the first element is paths, the second element is envsteps_this_batch, and the third element is train_video_paths
            # these are the trajectories we collected to be used for training
            self.total_envsteps += envsteps_this_batch
            # envsteps are the number of steps in the environment, which is the sum of the number of steps in each episode

            # relabel the collected obs with actions from a provided expert policy
            # this is the dagger part, where we relabel the data with the expert policy
            if relabel_with_expert and itr >= start_relabel_with_expert:
                # itr is the current iteration
                # start_relabel_with_expert is the iteration at which we want to start relabelling with the expert
                paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below
                # paths is a list of trajectories, where each trajectory is a dictionary with keys "observation", "action", "reward", "next_observation", "terminal"
                # description of the keys: observation is the observation at each time step, action is the action at each time step, reward is the reward at each time step, next_observation is the observation at the next time step, terminal is whether the episode is over

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)
            # replay buffer is a list of transitions, where each transition is a tuple (observation, action, reward, next_observation, terminal)
            # transitions are the data we use to train the agent

            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # HW1: implement this function below
            # we train the agent using the data we collected from the environment
            # that is we use the transitions in the replay buffer to train the agent

            # log/save
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy,
            batch_size,
    ):
        """
        :param itr: iteration numbers
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```        
        if itr == 0 and load_initial_expertdata is not None:
            with open(load_initial_expertdata, 'rb') as f:
                # f is a file object
                # rb is the mode, where r stands for read and b stands for binary
                loaded_paths = pickle.load(f)
                # pickle.load reads the data from the file and deserializes it
                # this is necessary because the data is stored in a binary format
                # otherwise, we would have to read the data as a string and then convert it to a list of dictionaries
                # pickle is a module that serializes and deserializes data, used to save and load data
            return loaded_paths, 0, None

        # (2) collect `self.params['batch_size']` transitions
        # this is done by interacting with the environment using the collect_policy

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        # sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array'))
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])
        # the difference between collect and expert policy is that the collect policy is the policy we are learning, the expert policy is the policy we are trying to imitate
        # we re using batch_size as the min_timesteps_per_batch and we re using self.params['ep_len'] as the max_path_length

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, self.MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            # sample(self, batch_size) comes from the agent class
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            # train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n) comes from the agent class
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")
        # doing this relabelling is the same as doing dagger, where we use the expert policy to relabel the data

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels

        for path in paths:
            # get action from the expert policy
            # get_action(self, obs) comes from the policy class
            path["action"] = expert_policy.get_action(path["observation"])
            # paths[i]["observation"] is the observation at each time step
            # paths[i]["action"] is the action at each time step
            # paths[i]["reward"] is the reward at each time step
            # paths[i]["next_observation"] is the observation at the next time step
            # paths[i]["terminal"] is whether the episode is over

        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
        # perform logging is done for each iteration, where we log the metrics and save the videos

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, self.MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)

            if itr == 0:
                # when we first start training, we log the initial return
                # this is the return we get from the initial policy, before we start training
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
            # this line is necessary to make sure that the data is written to the file