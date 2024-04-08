import argparse
import glob
import os
import re
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
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    q4_dirs = sorted([d for d in os.listdir(args.logdir) if d.startswith('hw4_q4')])
    q4_data = [os.path.join(args.logdir, d) for d in q4_dirs]
    q4_horizon = [glob.glob(os.path.join(d, 'events*'))[0] for d in q4_data if 'horizon' in d]
    print("q4_horizon: ", len(q4_horizon))
    q4_ensemble = [glob.glob(os.path.join(d, 'events*'))[0] for d in q4_data if 'ensemble' in d]
    print("q4_ensemble: ", len(q4_ensemble))
    q4_numseq = [glob.glob(os.path.join(d, 'events*'))[0] for d in q4_data if 'numseq' in d]
    print("q4_numseq: ", len(q4_numseq))

    all_X_horizon = []
    all_Y_horizon = []
    all_X_ensemble = []
    all_Y_ensemble = []
    all_X_numseq = []
    all_Y_numseq = []

    for eventfile in q4_horizon:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_horizon.append(X)
        all_Y_horizon.append(Y)
    
    print("Saving all_X_horizon: ", len(all_X_horizon))

    for eventfile in q4_ensemble:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_ensemble.append(X)
        all_Y_ensemble.append(Y)
    
    print("Saving all_X_ensemble: ", len(all_X_ensemble))

    for eventfile in q4_numseq:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_numseq.append(X)
        all_Y_numseq.append(Y)
    
    print("Saving all_X_numseq: ", len(all_X_numseq))

    horizon_pattern = "horizon\d+"
    horizon_legends = sorted([re.search(horizon_pattern, d).group() for d in q4_data if 'horizon' in d])
    
    ensemble_pattern = "ensemble\d+"
    ensemble_legends = sorted([re.search(ensemble_pattern, d).group() for d in q4_data if 'ensemble' in d])
    
    numseq_pattern = "numseq\d+"
    numseq_legends = [re.search(numseq_pattern, d).group() for d in q4_data if 'numseq' in d]
    
    print("Plotting")
    plt.figure()
    plt.title('Effect of the planning horizon')
    for i, (X, Y) in enumerate(zip(all_X_horizon, all_Y_horizon)):
        plt.plot(X, Y, label=horizon_legends[i])
    plt.xlabel('Train steps')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('q4_horizon.png')

    plt.figure()
    plt.title('Effect of the ensemble size')
    for i, (X, Y) in enumerate(zip(all_X_ensemble, all_Y_ensemble)):
        plt.plot(X, Y, label=ensemble_legends[i])
    plt.xlabel('Train steps')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('q4_ensemble.png')

    plt.figure()
    plt.title('Effect of the number of candidate action sequences')
    for i, (X, Y) in enumerate(zip(all_X_numseq, all_Y_numseq)):
        plt.plot(X, Y, label=numseq_legends[i])
    plt.xlabel('Train steps')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('q4_numseq.png')
    
    q5_dirs = sorted([d for d in os.listdir(args.logdir) if d.startswith('hw4_q5')])
    q5_data = [os.path.join(args.logdir, d) for d in q5_dirs]
    q5_events = [glob.glob(os.path.join(d, 'events*'))[0] for d in q5_data]
    
    all_X_q5 = []
    all_Y_q5 = []
    for eventfile in q5_events:
        X, Y = get_section_results(eventfile)
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        all_X_q5.append(X)
        all_Y_q5.append(Y)
    
    cem_pattern = "cem_\d+"
    random_pattern = "random"
    cem_legends = sorted([re.search(cem_pattern, d).group() for d in q5_data if 'cem' in d])
    random_legend = sorted([re.search(random_pattern, d).group() for d in q5_data if 'random' in d])
    q5_legends = cem_legends + random_legend
    
    plt.figure()
    plt.title('Comparison of CEM and Random-shooting methods')
    for i, (X, Y) in enumerate(zip(all_X_q5, all_Y_q5)):
        plt.plot(X, Y, label=q5_legends[i])
    plt.xlabel('Train steps')
    plt.ylabel('Return')
    plt.legend()
    plt.savefig('q5.png')

    # logdir = os.path.join(args.logdir, 'events*')
    # print(logdir)
    # eventfile = glob.glob(logdir)[0]

    # X, Y = get_section_results(eventfile)
    # for i, (x, y) in enumerate(zip(X, Y)):
    #     print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))
