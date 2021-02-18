# #!/usr/bin/env python3
# # file: phy_env_class.py
#
# GENERAL DEFINITIONS
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from collections import OrderedDict

from phy.phy_env_class import param_under_test


cw_num = 120


def command_line_parse():
    """
    Parse command line using arg-parse and get user data to run the plotting.

    :return: the parsed arguments: csv path, list of modes tto plots
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("-modes", nargs='+', type=int, required=False)
    args: dict = vars(parser.parse_args())
    return args["csv_path"], args["modes"]


def all_of_my_plots(csv_path, modes):
    # Use TeX Interpreter
    rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    save_path = os.path.join(csv_path, 'plots')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass

    # Simulation parameters
    mode = [[0, 1], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [1, 0]]
    modes = np.arange(5)

    # Read csv values for each algorithm
    opt_df = pd.read_csv(csv_path + f'/opt.csv')
    agg_df = pd.read_csv(csv_path + f'/aggressive.csv')
    ran_df = pd.read_csv(csv_path + f'/random.csv')
    tp_df = pd.read_csv(csv_path + f'/TP.csv')
    sma_df = pd.read_csv(csv_path + f'/smart.csv')
    rle_df = pd.read_csv(csv_path + f'/phy_experiment.csv')

    p = 0.5

    agg_mode = agg_df.loc[agg_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]
    # ran_mode = ran_df.loc[agg_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]
    tp_mode = tp_df.loc[agg_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]
    # opt_mode = opt_df.loc[opt_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]
    sma_mode = sma_df.loc[agg_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]
    rle_mode = rle_df.loc[agg_df['param_under_test'] == p].loc[agg_df['cw_classes'] != 0]


    # Reward plot
    fig, ax = plt.subplots()
    # plt.plot(modes, 1 - np.array(ran_mode['avg_embb_outages'].to_list())/120, c='blue', marker='o', linewidth=0.5, markersize=6, label=f'random')
    plt.plot(modes, np.array(agg_mode['avg_embb_outages'].to_list())/cw_num, c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
    # plt.plot(modes, np.array(opt_mode['avg_embb_outages'].to_list())/cw_num, c='orange', marker='^', linewidth=0.5, markersize=6, label=f'opt')
    plt.plot(modes, np.array(sma_mode['avg_embb_outages'].to_list())/cw_num, c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
    plt.plot(modes, np.array(tp_mode['avg_embb_outages'].to_list())/cw_num, c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
    plt.plot(modes, np.array(rle_mode['avg_embb_outages'].to_list())/cw_num, c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
    # Label
    ax.set_xlabel('$D$')
    ax.set_ylabel('Percentage of eMBB codeword in outage')
    # X axis
    plt.xticks([])
    plt.xticks(modes, ('[0, 1]', '[0.2, 0.8]', '[0.5, 0.5]', '[0.8, 0.2]', '[1, 0]'))
    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    # Show
    plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
    plt.savefig(save_path + f'/embb_outage_modes.pdf', format='pdf', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    results_path, modes_list = command_line_parse()
    all_of_my_plots(results_path, modes_list)
