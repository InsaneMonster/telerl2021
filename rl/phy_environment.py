# Import packages
import datetime
import logging
import numpy as np
import os

# Import usienarl
from usienarl import Environment, SpaceType

# Import phy
from phy import phy_env_class


class PhyEnvironment(Environment):
    """
    PHY task environment.
    """

    def __init__(self,
                 name: str,
                 rl_env_ver: str,
                 urllc_param: None or float,
                 vision_ahead: int = 0,
                 cw_class: list or None = [0, 1],
                 q_norm: float = 0,
                 render: bool = True,
                 random_queue: bool = False,
                 cw_tot_number: int = None):
        """Constructor of PhyEnvironment class

        Parameters
        __________
        :param name: str, label used by the agent.
        :param rl_env_ver: str, specify the version of the model. Until now, only 'bernoulli' is available.
        :param urllc_param: float, it represent the probability of activation of the Bernoulli
            distribution used to define if an urllc packet is arrived in queue at each step.
        :param cw_class: list, each element position represent the kind of class, the number
            represents the percentage of each class of codeword in the coherence time;
            e.g. if cw_class=[0,0.5,0.5] there will be 0% of class 0, 50% of class 1 and 50% of class 2.
        :param q_norm: float, normalization factor used to multiply the queue length in the reward.
        :param render: bool, if True will render each episode in a separated .txt file.
        """
        # Define environment attributes
        self._phy_environments: [] = []
        # Define activation probability
        self._urllc_param: float = urllc_param
        # Define version of the environment
        self._rl_env_ver: str = rl_env_ver
        # Define how many future steps are seen by the agent
        self._vision_ahead: int = vision_ahead
        # Define the codeword classes parameters
        self._cw_class = cw_class
        self._cw_tot_number = cw_tot_number
        # Define the normalization factor of the queue
        self._q_norm = q_norm
        # Queue is randomized
        self._random_queue = random_queue
        # Define if the environment should render
        self._render = render
        # Define environment empty attributes
        self._last_step_episode_done_flags: np.ndarray or None = None
        self._last_step_states: np.ndarray or None = None
        # Metrics
        self._embb_outage_counters_episode: [] = []
        self._embb_outage_counters: [] = []
        self._urllc_delay_counters_episode: [] = []
        self._urllc_delay_counters: [] = []
        self._residual_urllc_pkts_episode: [] = []
        self._residual_urllc_pkts: [] = []
        # Generate the base environment
        super(PhyEnvironment, self).__init__(name)

    def _generate(self,
                  logger: logging.Logger) -> bool:
        # Close all previous environments, if any
        self.close(logger, None)
        # Define render path for current experiment
        _render_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 "../phy_results/bernoulli/render"),
                                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        # Generate all new parallel environments
        for i in range(self._parallel):
            self._phy_environments.append(phy_env_class.Phy(*phy_env_class.std_param(self._urllc_param),
                                                            rl_ver=self._rl_env_ver,
                                                            vision_ahead=self._vision_ahead,
                                                            cw_tot_number=self._cw_tot_number,
                                                            cw_class_prob=self._cw_class,
                                                            q_norm=self._q_norm,
                                                            env_name='phy_name_' + str(i),
                                                            render=self._render, render_path=_render_path,
                                                            random_queue=self._random_queue))
        # Setup attributes
        self._last_step_episode_done_flags = np.zeros(self._parallel, dtype=bool)
        if self.state_space_type == SpaceType.continuous:
            self._last_step_states: np.ndarray = np.zeros((self._parallel, *self.state_space_shape), dtype=float)
        else:
            self._last_step_states: np.ndarray = np.zeros(self._parallel, dtype=int)
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        # Initialize cell and base station
        for i in range(len(self._phy_environments)):
            self._phy_environments[i].env_init()

    def close(self,
              logger: logging.Logger,
              session):
        # Clear all the environments
        self._phy_environments = []

    def reset(self,
              logger: logging.Logger,
              session) -> np.ndarray:
        # Prepare list of return values
        start_states: [] = []
        # Reset all parallel environments
        self._last_step_episode_done_flags = np.zeros(self._parallel, dtype=bool)
        if self.state_space_type == SpaceType.continuous:
            self._last_step_states: np.ndarray = np.zeros((self._parallel, *self.state_space_shape), dtype=float)
        else:
            self._last_step_states: np.ndarray = np.zeros(self._parallel, dtype=int)
        for i in range(len(self._phy_environments)):
            start_states.append(self._phy_environments[i].env_reset())
        # Reset metrics
        self._embb_outage_counters_episode = []
        self._urllc_delay_counters_episode = []
        self._residual_urllc_pkts_episode = []
        # Return start states wrapped in a numpy array
        return np.array(start_states)

    def step(self,
             logger: logging.Logger,
             session,
             action: np.ndarray) -> ():
        # Make sure the action is properly sized
        assert (len(self._phy_environments) == action.shape[0])
        # Prepare list of return values
        states: [] = []
        rewards: [] = []
        episode_done_flags: [] = []
        # Make a step in all non completed environments
        for i in range(len(self._phy_environments)):
            # Add dummy values to return if this parallel environment episode
            # is already done
            if self._last_step_episode_done_flags[i]:
                states.append(self._last_step_states[i])
                rewards.append(0.0)
                episode_done_flags.append(True)
                continue
            # Execute the step in this parallel environment
            state_next, reward, episode_done = self._phy_environments[i].env_step(action[i])
            # Save results
            states.append(state_next)
            rewards.append(reward)
            episode_done_flags.append(episode_done)
            # Save metrics
            if episode_done:
                self._embb_outage_counters_episode.append(self._phy_environments[i].embb_outage_counter)
                self._urllc_delay_counters_episode.append(self._phy_environments[i].urllc_delay_counter)
                self._residual_urllc_pkts_episode.append(self._phy_environments[i].residual_urllc_pkt)
            # Update last step flags and states
            self._last_step_episode_done_flags[i] = episode_done
            self._last_step_states[i] = state_next
        # Save metric
        if np.all(self._last_step_episode_done_flags):
            self._embb_outage_counters.append(np.mean(np.array(self._embb_outage_counters_episode)))
            self._urllc_delay_counters.append(np.mean(np.array(self._urllc_delay_counters_episode)))
            self._residual_urllc_pkts.append(np.mean(np.array(self._residual_urllc_pkts_episode)))
        # Return new states, rewards and episode done flags wrapped in np array
        return np.array(states), np.array(rewards), np.array(episode_done_flags)

    def render(self,
               logger: logging.Logger,
               session):
        # Make sure there is at least a parallel environment
        assert (len(self._phy_environments) > 0)
        # Render all the environments
        for i in range(len(self._phy_environments)):
            self._phy_environments[i].env_render()

    def sample_action(self,
                      logger: logging.Logger,
                      session) -> np.ndarray:
        # Prepare list of return values
        actions: [] = []
        # Sample action from all parallel environment
        for i in range(len(self._phy_environments)):
            actions.append(self._phy_environments[i].env_sample_action())
        # Return sampled actions wrapped in numpy array
        return np.array(actions)

    def possible_actions(self,
                         logger: logging.Logger,
                         session) -> []:
        # Compute the possible actions for all parallel environments
        possible_actions: [] = []
        for i in range(len(self._phy_environments)):
            if self._phy_environments[i].urllc_state[0] > 0:
                possible_actions.append(list(self._phy_environments[i].action_space))
            else:
                possible_actions.append([0])
        # Return the possible actions
        return possible_actions

    @property
    def state_space_type(self) -> SpaceType:
        return SpaceType.continuous

    @property
    def state_space_shape(self) -> ():
        assert (len(self._phy_environments) > 0)
        # Note: they are all equal so the first is used
        return self._phy_environments[0].env_get_state(only_shape=True)

    @property
    def action_space_type(self) -> SpaceType:
        return SpaceType.discrete

    @property
    def action_space_shape(self) -> ():
        assert (len(self._phy_environments) > 0)
        # Note: they are all equal so the first is used
        return self._phy_environments[0].action_space.shape

    def rng(self, i):
        return self._phy_environments[i].rng

    # Internal parameters
    @property
    def urllc_param(self):
        return self._urllc_param

    @property
    def cw_class(self):
        return self._cw_class

    @property
    def q_norm(self):
        return self._q_norm

    @property
    def urllc_state_range(self):
        # Note: they are all equal so the first is used
        return self._phy_environments[0].urllc_state_range

    @property
    def embb_state_range(self):
        # Note: they are all equal so the first is used
        return self._phy_environments[0].embb_state_range

    @property
    def embb_state_range_present(self):
        # Note: they are all equal so the first is used
        return self._phy_environments[0].embb_state_range_present

    @property
    def vision_ahead(self):
        return self._vision_ahead

    @property
    def frequency_resources(self):
        # Note: they are all equal so the first is used
        return self._phy_environments[0].cluster.subs

    # Metrics
    @property
    def embb_outage_counters(self):
        return self._embb_outage_counters

    @property
    def urllc_delay_counters(self):
        return self._urllc_delay_counters

    @property
    def residual_urllc_pkts(self):
        return self._residual_urllc_pkts
