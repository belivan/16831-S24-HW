from rob831.infrastructure.utils import *


class ReplayBuffer(object):

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        # concatenated means that we concatenate the data from each rollout into a single array
        self.obs = None
        self.acs = None
        self.rews = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):  # this method returns the length of the replay buffer
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):
        # as single rollout is a dictionary with keys as 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))
        # concat_rew is a boolean variable that is True by default, which means that the rewards are concatenated

        if self.obs is None:
            # if the replay buffer is empty, then we just add the new rollouts
            self.obs = observations[-self.max_size:]  # -self.max_size: means that we take the last max_size number of elements
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            # here we concatenate the new rollouts with the existing rollouts, and take the last max_size number of elements
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                # if the rewards are not concatenated, then we add the rewards as a list
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    # if the rewards are not a list, then we add the rewards as a single element
                    self.rews.append(rewards)
                # we take the last max_size number of elements, this is done to ensure that the length of the rewards
                # is not greater than max_size
                self.rews = self.rews[-self.max_size:]
            # next_observations and terminals are concatenated in the same way as observations and actions
            # next_obs are used to store the next observations, this is used to calculate the Q-values???
            # terminals are used to store the terminal states, this is used to calculate the Q-values???
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):  # this method samples random data from the replay buffer
        # this data is useful for training the policy, which is done in the update method of the policy???

        # the following assert statement checks if the length of the observations, actions, rewards, next_observations, and terminals are the same
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        ## TODO return batch_size number of random entries from each of the 5 component arrays above [OK]
        ## HINT 1: use np.random.permutation to sample random indices
        # np.random.permutation generates a random permutation of the indices, permutation means that the order of the indices is changed
        ## HINT 2: return corresponding data points from each array (i.e., not different indices from each array)
        ## HINT 3: look at the sample_recent_data function below
        r_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return (
            # return batch_size number of random entries from each of the 5 component arrays
            # code explanation:
            # np.random.permutation(self.obs.shape[0])[:batch_size] generates a random permutation of the indices of the observations
            # [:batch_size] takes the first batch_size number of elements from the random permutation
            # self.obs[np.random.permutation(self.obs.shape[0])[:batch_size]] returns the observations at the random indices
            self.obs[r_indices],
            self.acs[r_indices],
            self.rews[r_indices],
            self.next_obs[r_indices],
            self.terminals[r_indices],
        )

    def sample_recent_data(self, batch_size=1):
        return (
            # return the last batch_size number of elements from each of the component arrays
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )
