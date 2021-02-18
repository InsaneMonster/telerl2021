# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType
from usienarl.utils import run_experiment, command_line_parse
from usienarl.models import ProximalPolicyOptimization
from usienarl.agents import PPOAgent

# Import required src

from rl.phy_environment import PhyEnvironment
from rl.phy_experiment import PhyExperiment

# Define utility functions to run the experiment


def _define_ppo_model(actor_config: Config, critic_config: Config) -> ProximalPolicyOptimization:
    # Define attributes
    learning_rate_policy: float = 3e-5
    learning_rate_advantage: float = 1e-4
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    policy_steps_per_update: int = 80
    minibatch_size: int = 32
    lambda_parameter: float = 0.97
    clip_ratio: float = 0.2
    target_kl_divergence: float = 0.01
    # Return the model
    return ProximalPolicyOptimization("model", actor_config, critic_config,
                                      discount_factor,
                                      learning_rate_policy, learning_rate_advantage,
                                      value_steps_per_update, policy_steps_per_update,
                                      minibatch_size,
                                      lambda_parameter,
                                      clip_ratio,
                                      target_kl_divergence)


def _define_agent(model: ProximalPolicyOptimization) -> PPOAgent:
    # Define attributes
    update_every_episodes: int = 1000
    # Return the agent
    return PPOAgent("ppo_agent", model, update_every_episodes)


def run(workspace: str,
        iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Generate the Phy Environment
    environment: PhyEnvironment = PhyEnvironment("phy_environment", urllc_param=None, rl_env_ver='bernoulli', vision_ahead=6,
                                                 cw_tot_number=120, cw_class=None, q_norm=0)
    # Note: thresholds are high so we can perform training without worrying about reaching a certain goal
    validation_threshold: float = 0.0
    validation_std: float or None = None
    test_threshold: float = 0.0
    test_std: float or None = None
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [128, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_1")
    nn_config.add_hidden_layer(LayerType.dense, [64, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_2")
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_3")

    # Define model
    inner_model: ProximalPolicyOptimization = _define_ppo_model(actor_config=nn_config, critic_config=nn_config)
    # Define agent
    ppo_agent: PPOAgent = _define_agent(inner_model)
    # Define experiment
    experiment: PhyExperiment = PhyExperiment("phy_experiment-6",
                                              validation_threshold=validation_threshold, validation_std=validation_std,
                                              test_threshold=test_threshold, test_std=test_std,
                                              environment=environment, agent=ppo_agent)
    # Define experiment data
    saves_to_keep: int = 15
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 2000
    validation_episodes: int = 100
    training_validation_volleys: int = 30
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 140
    # Run experiment
    run_experiment(logger=logger, experiment=experiment,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run this experiment
    run(workspace_path, experiment_iterations, render_during_training, render_during_validation, render_during_test)
