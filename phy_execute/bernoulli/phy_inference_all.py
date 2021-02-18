# Import packages

import os
import argparse

# Import required src
from phy.phy_env_class import mode_classes
from phy.phy_env_class import param_under_test

from phy_execute.bernoulli.phy_ppo_inference import run as run_ppo
from phy_execute.bernoulli.phy_aggressive_inference import run as run_aggressive
from phy_execute.bernoulli.phy_random_inference import run as run_random
from phy_execute.bernoulli.phy_tp_lazy_inference import run as run_tp_lazy
from phy_execute.bernoulli.phy_opt_inference import run as run_opt
from phy_execute.bernoulli.phy_tp_inference import run as run_tp
from phy_results.plotting_relevant_metrics_paper import all_of_my_plots


def command_line_parse():
    """
    Parse command line using arg-parse and get user data to run the experiment in inference mode.
    Note: model iteration is supposed to be -1 when not available.

    :return: the parsed arguments: restore path of the model checkpoint, log path for the test, CUDA devices, model iteration and optional render flag
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("restore_path", type=str)
    parser.add_argument("log_path", type=str)
    parser.add_argument("CUDA_devices", type=str)
    parser.add_argument("iteration", type=int)
    parser.add_argument("-render", action="store_true")
    parser.add_argument("-cw_number", type=int, default=120)
    parser.add_argument("-csv_path", default=None)
    parser.add_argument("-episodes", default=1000, type=int)
    args: dict = vars(parser.parse_args())
    return args["restore_path"], args["log_path"], args["CUDA_devices"], args["iteration"], args["render"], \
           args["cw_number"], args["csv_path"], args["episodes"]


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    restore_path, log_path, cuda_devices, experiment_iteration_number, render, cw_number, csv_path, episodes = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run all phy inference
    modes = [0, 1, 2, 3, 4, 5]
    for mode in modes:
        for p in param_under_test:
            run_ppo(restore_path, log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
            run_opt(log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
            run_aggressive(log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
            run_random(log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
            run_tp_lazy(log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
            run_tp(log_path, experiment_iteration_number, render, p, cw_number, mode_classes[mode], csv_path, episodes)
    # Plot metrics
    all_of_my_plots(csv_path, modes, cw_number)
