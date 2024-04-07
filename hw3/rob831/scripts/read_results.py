import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

'''Submit the run logs (in rob831/data) for all of the experiments above. In your report, make a single graph that
averages the performance across three runs for both DQN and double DQN. See scripts/read results.py
for an example of how to read the evaluation returns from Tensorboard logs.'''

def get_section_results(file):  # this function is not used in the code, but it is useful to read the results from the tensorboard
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    q1_doubledqn = [os.path.join(args.logdir, d) for d in os.listdir(args.logdir) if d.startswith('q1_doubledqn')]
    q1_doubledqn_events = [glob.glob(os.path.join(d, 'events*'))[0] for d in q1_doubledqn]

    q1_dqn = [os.path.join(args.logdir, d) for d in os.listdir(args.logdir) if d.startswith('q1_dqn')]
    q1_dqn_events = [glob.glob(os.path.join(d, 'events*'))[0] for d in q1_dqn]

    all_X_dqn = []
    all_Y_dqn = []

    all_X_doubledqn = []
    all_Y_doubledqn = []

    for eventfile in q1_dqn_events:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_dqn.append(X)
        all_Y_dqn.append(Y)

    for eventfile in q1_doubledqn_events:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_doubledqn.append(X)
        all_Y_doubledqn.append(Y)

    X_dqn = all_X_dqn[0]
    X_doubledqn = all_X_doubledqn[0]

    Y_dqn = np.mean(all_Y_dqn, axis=0)
    Y_doubledqn = np.mean(all_Y_doubledqn, axis=0)

    Y_dqn_error = []
    Y_doubledqn_error = []

    Y_dqn_std = np.std(all_Y_dqn, axis=0)
    Y_doubledqn_std = np.std(all_Y_doubledqn, axis=0)

    plt.figure()
    plt.title('DQN vs Double DQN (Average of 3 runs)')
    plt.errorbar(X_dqn, Y_dqn, yerr=Y_dqn_std, fmt='b', label='DQN w/ standard error bars')
    plt.errorbar(X_doubledqn, Y_doubledqn, yerr=Y_doubledqn_std, fmt='r', label='Double DQN w/ standard error bars')
    # plt.plot(X_dqn, Y_dqn, 'b', label='DQN')
    # plt.plot(X_doubledqn, Y_doubledqn, 'r', label='Double DQN')
    plt.xlabel('Train steps')
    plt.ylabel('Return')
    plt.legend()
    plt.show()

    # logdir = os.path.join(args.logdir, 'events*')
    # print(logdir)
    # eventfile = glob.glob(logdir)[0]

    # X, Y = get_section_results(eventfile)
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
