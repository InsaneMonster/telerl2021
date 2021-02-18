# Import packages

import logging
import numpy as np
import pandas as pd
import os

# Import usienarl

from usienarl import Experiment, Agent, Interface

# Import requires src

from rl.phy_environment import PhyEnvironment


class PhyExperiment(Experiment):
    """
    PHY task Experiment.

    It uses a validation threshold and a test threshold.
    """

    def __init__(self,
                 name: str,
                 validation_threshold: float,
                 validation_std: float or None,
                 test_threshold: float,
                 test_std: float or None,
                 environment: PhyEnvironment,
                 agent: Agent,
                 interface: Interface = None,
                 csv_path: str = None,
                 model: int = 0):
        # Generate the base experiment
        super(PhyExperiment, self).__init__(name, environment, agent, interface)
        # Define internal attributes
        self._validation_threshold: float = validation_threshold
        self._validation_std: float = validation_std
        self._test_threshold: float = test_threshold
        self._test_std: float = test_std
        self._csv_path = csv_path
        self._model = model

    def _is_validated(self,
                      logger: logging.Logger) -> bool:
        # Print internal metrics
        self._print_metrics(logger)
        # Get all step rewards, episode total and scaled rewards and episode lengths of last training volley
        # last_training_volley_step_rewards: [] = self.training_volley.rewards
        # last_training_volley_episode_total_rewards: [] = self.training_volley.total_rewards
        # last_training_volley_episode_scaled_rewards: [] = self.training_volley.scaled_rewards
        # last_training_volley_episode_episode_lengths: [] = self.training_volley.episode_lengths
        # Get all step rewards, episode total and scaled rewards and episode lengths of all training volleys (indexed by training volley number)
        # all_training_volleys_step_rewards: [] = self.training_rewards
        # all_training_volleys_episode_total_rewards: [] = self.training_total_rewards
        # all_training_volleys_episode_scaled_rewards: [] = self.training_scaled_rewards
        # all_training_volleys_episode_episode_lengths: [] = self.training_episode_lengths
        # Get all step rewards, episode total and scaled rewards and episode lengths of last validation volley
        # last_validation_volley_step_rewards: [] = self.validation_volley.rewards
        # last_validation_volley_episode_total_rewards: [] = self.validation_volley.total_rewards
        # last_validation_volley_episode_scaled_rewards: [] = self.validation_volley.scaled_rewards
        # last_validation_volley_episode_episode_lengths: [] = self.validation_volley.episode_lengths
        # Get all step rewards, episode total and scaled rewards and episode lengths of all validation volleys (indexed by validation volley number)
        # all_validation_volleys_step_rewards: [] = self.validation_rewards
        # all_validation_volleys_episode_total_rewards: [] = self.validation_total_rewards
        # all_validation_volleys_episode_scaled_rewards: [] = self.validation_scaled_rewards
        all_validation_volleys_episode_episode_lengths: [] = self.validation_episode_lengths
        # Check if average validation total reward on the last validation
        # volley is greater than validation threshold
        if self.validation_volley.avg_total_reward >= self._validation_threshold:
            # Standard deviation threshold could be not defined
            if self._validation_std is None:
                return True
            # Also check standard deviation to be below the given threshold
            if self.validation_volley.std_total_reward <= self._validation_std:
                return True
        return False

    def _is_successful(self,
                       logger: logging.Logger) -> bool:
        # Print internal metrics
        self._print_metrics(logger)
        # Get all step rewards, episode total and scaled rewards and episode lengths of last test volley
        # last_test_volley_step_rewards: [] = self.test_volley.rewards
        # last_test_volley_episode_total_rewards: [] = self.test_volley.total_rewards
        # last_test_volley_episode_scaled_rewards: [] = self.test_volley.scaled_rewards
        # last_test_volley_episode_episode_lengths: [] = self.test_volley.episode_lengths
        # Get all step rewards, episode total and scaled rewards and episode lengths of all test volleys (indexed by test volley number)
        # all_test_volleys_step_rewards: [] = self.test_rewards
        # all_test_volleys_episode_total_rewards: [] = self.test_total_rewards
        # all_test_volleys_episode_scaled_rewards: [] = self.test_scaled_rewards
        # all_test_volleys_episode_episode_lengths: [] = self.test_episode_lengths
        # Check if average test total reward over all test volleys is greater
        # than test threshold
        if self.avg_test_avg_total_reward >= self._test_threshold:
            # Standard deviation threshold could be not defined
            if self._test_std is None:
                return True
            # Also check standard deviation to be below the given threshold
            if self.avg_test_std_total_reward <= self._test_std:
                return True
        return False

    def _print_metrics(self,
                       logger: logging.Logger):
        # Compile the logger
        logger.info(f"Average embb outage counter {np.mean(np.array(self.environment.embb_outage_counters))}")
        logger.info(f"Average urllc latency violated {np.mean(np.array(self.environment.urllc_delay_counters))}")
        logger.info(f"Average residual urllc packets {np.mean(np.array(self.environment.residual_urllc_pkts))}")
        # If the csv path is specified, the csv file is printed
        if self._csv_path is not None:
            cw_class_dict = {'None': 0, '[0, 1]': 1, '[0.2, 0.8]': 2, '[0.5, 0.5]': 3, '[0.8, 0.2]': 4, '[1, 0]': 5}
            if not os.path.isdir(self._csv_path):
                try:
                    os.makedirs(self._csv_path)
                except FileExistsError:
                    pass
            output_path = os.path.join(self._csv_path, self.name + '.csv')
            label = {'model': self._model,
                     'param_under_test': self.environment.urllc_param,
                     'cw_classes': cw_class_dict[str(self.environment.cw_class)],
                     'q_norm': self.environment.q_norm,
                     'avg_tot_reward': self.avg_test_avg_total_reward,
                     'avg_steps': self.avg_test_avg_episode_length,
                     'avg_embb_outages': np.mean(np.array(self.environment.embb_outage_counters)),
                     'avg_urllc_delays': np.mean(np.array(self.environment.urllc_delay_counters)),
                     'avg_residual_urllc_pkts': np.mean(np.array(self.environment.residual_urllc_pkts)),
                     'avg_action_duration': self.avg_test_avg_action_duration}
            df = pd.DataFrame(label, index=[0])
            # saving the data frame
            df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
