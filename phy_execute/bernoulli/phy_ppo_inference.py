# Import packages

import tensorflow
import logging
import os
import argparse
import sys
import numpy as np

# Import usienarl

from usienarl import Config, LayerType
from usienarl.models import ProximalPolicyOptimization
from usienarl.agents import PPOAgent

# Import required src

from rl.phy_environment import PhyEnvironment
from rl.phy_experiment import PhyExperiment
from phy.phy_env_class import mode_classes
from phy.phy_env_class import param_under_test

# Define utility functions to run the experiment in inference mode only (test)


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
    parser.add_argument("-csv_path", default=None)
    args: dict = vars(parser.parse_args())
    return args["restore_path"], args["log_path"], args["CUDA_devices"], args["iteration"], args["render"], args["csv_path"]


def _define_ppo_model(actor_config: Config, critic_config: Config) -> ProximalPolicyOptimization:
    # Return the model (with default attributes for they are not relevant)
    return ProximalPolicyOptimization("model", actor_config, critic_config)


def _define_agent(model: ProximalPolicyOptimization) -> PPOAgent:
    # Return the agent (with default attributes for they are not relevant)
    return PPOAgent("ppo_agent", model)


def run(restore_path: str, log_path: str, iteration: int, render: bool,
        urllc_param: float, cw_tot_number: int, cw_classes: list,
        csv_path: str = None, episodes: int = 1000):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Generate the Phy Environment
    # Note: formally this should be changed each time to be like the environment the model was trained in
    # However it works anyway, so long that action and state spaces are the same
    environment: PhyEnvironment = PhyEnvironment("phy_environment", rl_env_ver='bernoulli', render=render,
                                                 urllc_param=urllc_param, q_norm=0,
                                                 cw_tot_number=cw_tot_number, cw_class=cw_classes)
    # Note: these values should be set according to the environment the model is tested in
    validation_threshold: float = 0.0
    validation_std: float or None = None
    test_threshold: float = 0.0
    test_std: float or None = None
    # Define Neural Network layers
    # Note: this is important, should be the same of the loaded model (the initializers are not relevant)
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [128, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_1")
    nn_config.add_hidden_layer(LayerType.dense, [64, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_2")
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_3")
    # Define model
    inner_model: ProximalPolicyOptimization = _define_ppo_model(actor_config=nn_config, critic_config=nn_config)
    # Define agent
    ppo_agent: PPOAgent = _define_agent(inner_model)
    # Define experiment
    experiment: PhyExperiment = PhyExperiment("phy_experiment",
                                              validation_threshold=validation_threshold, validation_std=validation_std,
                                              test_threshold=test_threshold, test_std=test_std,
                                              environment=environment, agent=ppo_agent,
                                              csv_path=csv_path, model=iteration)
    # Define experiment data
    # episodes: int = 100
    volleys: int = 1
    episode_length_max: int = 1400
    if experiment.setup(logger=logger, iteration=iteration):
        # Prepare the logger handlers
        logger.handlers = []
        # Generate a console and a file handler for the logger
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        file_handler: logging.FileHandler = logging.FileHandler(log_path + "/info.log", "w+")
        # Set handlers properties
        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        formatter: logging.Formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        # Add the handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        # Actually test the model
        experiment.test(logger=logger, episodes=episodes, volleys=volleys, episode_length=episode_length_max,
                        restore_path=restore_path, render=render)


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    model_restore_path, info_log_path, cuda_devices, model_iteration, render_flag, csv_path = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run this experiment
    cw_tot_num = 120
    param = param_under_test
    for mode in [0, 1, 2, 3, 4, 5]:
        for p in param:
            run(model_restore_path, info_log_path, model_iteration, render_flag, p, cw_tot_num, mode_classes[mode], csv_path)
