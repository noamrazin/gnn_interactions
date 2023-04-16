import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path
import os
import re
import json
import sys


TRAIN_VAL_TEST = 'test'
SUPPORTED_ALGS = ['random', 'spectral', 'ugs', 'wis', 'one_wis']
ALGS_NAMES = ['random', 'spectral', 'UGS', '2-WIS (ours)', '1-WIS (ours)']
COLORS = ['#EC5809', '#B85194', '#5B8F53', '#3F74CA', '#023A8D']
MARKERS = ["d", "o", "X", "^", "v", "s", "h", "1", "<", "*"]
LABELS_SIZE = 12
EXPERIMENTS_REGEX = r'([A-Za-z\-]+)_([a-zA-Z]+)_layers\d*_d\d*_edges_ratio_([.0-9\-]*)_.*iter(\d*)_([a-z_]+)_(pid.*|[0-9_]+)'
TITLE = None


def capital_title(s):
    return ' '.join([s_[0].upper() + s_[1:] for s_ in s.split(' ')])

def read_experiments_data(data_folder):
    p = re.compile(EXPERIMENTS_REGEX)
    experiments = os.listdir(data_folder)
    experiments_data = {}
    for ex in experiments:
        m = p.match(ex)
        if not m:
            continue
        dataset = m.group(1)
        model = m.group(2)
        alg = m.group(5)
        if 'train' in alg: # skip training algorithms
            continue
        ratio = '.'.join(m.group(3).split('-'))
        it = int(m.group(4))
        if not os.path.isfile(data_folder+ex+'/summary.json'):
            continue

        with open(data_folder+ex+'/summary.json', 'r') as f:
            data = json.load(f)
        if alg not in experiments_data:
            experiments_data[alg] = {}
        if ratio not in experiments_data[alg]:
            experiments_data[alg][ratio] = []
        experiments_data[alg][ratio].append(data)
    
    return experiments_data, dataset, model

def main(data_folder, output_folder):
    experiments_data, dataset, model = read_experiments_data(data_folder)
    
    results = {}
    results_stds = {}
    results_num = {}
    for alg in experiments_data:
        accuracies_data = {k: [data['best_score_epoch_tracked_values'][f'{TRAIN_VAL_TEST} accuracy']['value']
                               for data in experiments_data[alg][k]]
                           for k in experiments_data[alg]}
        means_ours = np.array([(float(k),np.mean(accuracies_data[k])) for k in sorted(accuracies_data.keys())])
        stds_ours = np.array([(float(k),np.std(accuracies_data[k])) for k in sorted(accuracies_data.keys())])
        size_ours = np.sum([len(accuracies_data[k]) for k in sorted(accuracies_data.keys())])
        results[alg] = means_ours
        results_stds[alg] = stds_ours
        results_num[alg] = size_ours

    fig = plt.figure()
    for i, alg in enumerate(SUPPORTED_ALGS):
        if alg not in experiments_data:
            continue
        print(ALGS_NAMES[i], f'(N={results_num[alg]})')
        plt.plot(100*(1-results[alg][:,0]), 100*(results[alg][:,1]), f'{MARKERS[i]}-', color=f'{COLORS[i]}', label=f'{ALGS_NAMES[i]}')
        plt.errorbar(100*(1-results[alg][:,0]), 100*results[alg][:,1],
                     100*results_stds[alg][:,1], color=f'{COLORS[i]}', linestyle='-',
                     alpha=0.2)


    plt.title(f'{capital_title(dataset)} ({model})' if not TITLE else TITLE)
    plt.xlabel('% of removed edges', fontsize=LABELS_SIZE)
    plt.ylabel('test accuracy (%)', fontsize=LABELS_SIZE)
    plt.legend()
    fig.set_size_inches(4.8, 2.8)
    fig.tight_layout()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    plt.savefig(f'{output_folder}/pruning_algorithms_comparison_{dataset}_{model}.png')
    plt.savefig(f'{output_folder}/pruning_algorithms_comparison_{dataset}_{model}.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="plot the results of different pruning algorithms produced by edge_removal_plan_runner.py")
    parser.add_argument('data_folder', help="folder with the outputs of edge_removal_plan_runner.py")
    parser.add_argument('output_folder', help="output folders for plots")
    args = parser.parse_args()
    main(args.data_folder + '/', args.output_folder)

