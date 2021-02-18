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


def command_line_parse():
    """
    Parse command line using arg-parse and get user data to run the plotting.

    :return: the parsed arguments: csv path, list of modes tto plots
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("-modes", nargs='+', type=int, required=True)
    args: dict = vars(parser.parse_args())
    return args["csv_path"], args["modes"]


def all_of_my_plots(csv_path, modes, cw_num=None):
    if cw_num is None:
        cw_num = 120
    # Use TeX Interpreter
    rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    save_path = os.path.join(csv_path, 'plots_letter')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass

    # Read csv values for each algorithm
    sma_df = pd.read_csv(csv_path + f'/TP-lazy.csv')
    tp_df = pd.read_csv(csv_path + f'/TP.csv')
    agg_df = pd.read_csv(csv_path + f'/aggressive.csv')
    try:
        rle_df = pd.read_csv(csv_path + f'/phy_experiment.csv')
    except FileNotFoundError:
        pass
    p = np.array(param_under_test)

    for mode in modes:
        # Table per mode
        agg_mode = agg_df.loc[agg_df['cw_classes'] == mode]
        sma_mode = sma_df.loc[sma_df['cw_classes'] == mode]
        tp_mode = tp_df.loc[tp_df['cw_classes'] == mode]
        try:
            rle_mode = rle_df.loc[rle_df['cw_classes'] == mode]            
        except NameError:
            pass


        # Reward plot
        fig, ax = plt.subplots()
        plt.plot(p, agg_mode['avg_tot_reward'].to_list()[:len(param_under_test)], c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
        plt.plot(p, sma_mode['avg_tot_reward'].to_list()[:len(param_under_test)], c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
        plt.plot(p, tp_mode['avg_tot_reward'].to_list()[:len(param_under_test)], c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
        try:
            plt.plot(p, rle_mode['avg_tot_reward'].to_list()[:len(param_under_test)], c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
        except NameError:
            pass
        # Label
        ax.set_xlabel('$p_u$')
        ax.set_ylabel('Average total reward')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Show
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.savefig(save_path + f'/avg_tot_reward_m{mode}_letter.pdf', format='pdf', dpi=300)
        plt.close(fig)

        # Steps plot
        fig, ax = plt.subplots()
        plt.plot(p, agg_mode['avg_steps'].to_list()[:len(param_under_test)], c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
        plt.plot(p, sma_mode['avg_steps'].to_list()[:len(param_under_test)], c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
        plt.plot(p, tp_mode['avg_steps'].to_list()[:len(param_under_test)], c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
        try:
            plt.plot(p, rle_mode['avg_steps'].to_list()[:len(param_under_test)], c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
        except NameError:
            pass
        # Label
        ax.set_xlabel('$p_u$')
        ax.set_ylabel('Average steps')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Show
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.savefig(save_path + f'/avg_steps_m{mode}.pdf', format='pdf', dpi=300)
        plt.close(fig)

        # embb_outages plot
        fig, ax = plt.subplots()
        plt.plot(p, agg_mode['avg_embb_outages'].to_numpy()[:len(param_under_test)]/cw_num, c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
        plt.plot(p, sma_mode['avg_embb_outages'].to_numpy()[:len(param_under_test)]/cw_num, c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
        plt.plot(p, tp_mode['avg_embb_outages'].to_numpy()[:len(param_under_test)]/cw_num, c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
        try:
            plt.plot(p, np.array(rle_mode['avg_embb_outages'].to_list()[:len(param_under_test)])/cw_num, c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
        except NameError:
            pass
        # Label
        ax.set_xlabel('$p_u$')
        ax.set_ylabel('Percentage of eMBB codeword in outage')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Show
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.savefig(save_path + f'/avg_embb_outages_m{mode}_letter.pdf', format='pdf', dpi=300)
        plt.close(fig)

        # urllc delays outages plot
        fig, ax = plt.subplots()
        plt.plot(p, agg_mode['avg_urllc_delays'].to_list()[:len(param_under_test)], c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
        plt.plot(p, sma_mode['avg_urllc_delays'].to_list()[:len(param_under_test)], c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
        plt.plot(p, tp_mode['avg_urllc_delays'].to_list()[:len(param_under_test)], c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
        try:
            plt.plot(p, rle_mode['avg_urllc_delays'].to_list()[:len(param_under_test)], c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
        except NameError:
            pass
        # Label
        ax.set_xlabel('$p_u$')
        ax.set_ylabel('Average percentage of URLLC packets violating the latency')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Show
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.savefig(save_path + f'/avg_urllc_delays_m{mode}_letter.pdf', format='pdf', dpi=300)
        plt.close(fig)

        # urllc delays outages plot
        fig, ax = plt.subplots()
        plt.plot(p, agg_mode['avg_residual_urllc_pkts'].to_list()[:len(param_under_test)], c='red', marker='x', linewidth=0.5, markersize=6, label=f'aggressive')
        plt.plot(p, sma_mode['avg_residual_urllc_pkts'].to_list()[:len(param_under_test)], c='grey', marker='d', linewidth=0.5, markersize=6, label=f'TP-lazy')
        plt.plot(p, tp_mode['avg_residual_urllc_pkts'].to_list()[:len(param_under_test)], c='purple', marker='v', linewidth=0.5, markersize=6, label=f'TP')
        try:
            plt.plot(p, rle_mode['avg_residual_urllc_pkts'].to_list()[:len(param_under_test)], c='black', linestyle='-.', marker='<', linewidth=0.5, label=f'PPO')
        except NameError:
            pass
        # Label
        ax.set_xlabel('$p_u$')
        ax.set_ylabel('Average number of URLLC packets remained in the queue')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Show
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.savefig(save_path + f'/avg_residual_m{mode}_letter.pdf', format='pdf', dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    results_path, modes_list = command_line_parse()
    all_of_my_plots(results_path, modes_list)
