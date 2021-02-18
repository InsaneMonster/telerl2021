#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import logging
import numpy

# Import usienarl

from usienarl import Agent, Interface, SpaceType


class TPLazyAgent(Agent):
    """
    Smart
    """

    def __init__(self,
                 name: str):
        # Generate base agent
        super(TPLazyAgent, self).__init__(name)

    def setup(self,
              logger: logging.Logger,
              scope: str,
              parallel: int,
              observation_space_type: SpaceType, observation_space_shape: (),
              agent_action_space_type: SpaceType, agent_action_space_shape: (),
              summary_path: str = None, save_path: str = None, saves_to_keep: int = 0) -> bool:
        # Make sure parameters are correct
        assert (parallel > 0)
        logger.info("Setup of agent " + self._name + " with scope " + scope + "...")
        # Reset agent attributes
        self._scope = scope
        self._parallel = parallel
        self._observation_space_type: SpaceType = observation_space_type
        self._observation_space_shape = observation_space_shape
        self._agent_action_space_type: SpaceType = agent_action_space_type
        self._agent_action_space_shape = agent_action_space_shape
        # Use a blank generate method
        if not self._generate(logger,
                              observation_space_type, observation_space_shape,
                              agent_action_space_type, agent_action_space_shape):
            return False
        # Validate setup
        return True

    def restore(self,
                logger: logging.Logger,
                session,
                path: str) -> bool:
        return True

    def save(self,
             logger: logging.Logger,
             session):
        pass

    def _generate(self,
                  logger: logging.Logger,
                  observation_space_type: SpaceType, observation_space_shape: (),
                  agent_action_space_type: SpaceType, agent_action_space_shape: ()) -> bool:
        return True

    def initialize(self,
                   logger: logging.Logger,
                   session):
        pass

    def act_warmup(self,
                   logger: logging.Logger,
                   session,
                   interface: Interface,
                   agent_observation_current: numpy.ndarray,
                   warmup_step: int, warmup_episode: int):
        pass

    def act_train(self,
                  logger: logging.Logger,
                  session,
                  interface: Interface,
                  agent_observation_current: numpy.ndarray,
                  train_step: int, train_episode: int):
        # Act as inference
        return self.act_inference(logger, session, interface, agent_observation_current, train_step, train_episode)

    def act_inference(self,
                      logger: logging.Logger,
                      session,
                      interface: Interface,
                      agent_observation_current: numpy.ndarray,
                      inference_step: int, inference_episode: int):
        # Choose wisely
        action = []
        for i in range(self.parallel):
            if agent_observation_current[i, interface.environment.urllc_state_range[0]] == 0:
                action.append(0)
            elif agent_observation_current[i, interface.environment.urllc_state_range[1]] <= 0:
                if len(interface.environment.possible_actions(logger, session)[i]) == 2:
                    current_action = 1
                else:
                    embb_state = agent_observation_current[i, interface.environment.embb_state_range_present]
                    current_action = numpy.argmax(embb_state) + 1
                action.append(current_action)
            else:
                embb_state = agent_observation_current[i, interface.environment.embb_state_range]
                present_state = embb_state[:len(embb_state) // (1 + interface.environment.vision_ahead)]
                future_states = embb_state[len(embb_state) // (1 + interface.environment.vision_ahead):].reshape(interface.environment.vision_ahead, interface.environment.frequency_resources)
                if numpy.any(sum(present_state) >= numpy.sum(future_states, axis=1)):
                    if len(interface.environment.possible_actions(logger, session)[i]) == 2:
                        current_action = 1
                    else:
                        current_action = numpy.argmax(present_state) + 1
                    action.append(current_action)
                else:
                    action.append(0)
        # Return the exploration action
        return numpy.array(action)

    def complete_step_warmup(self,
                             logger: logging.Logger,
                             session,
                             interface: Interface,
                             agent_observation_current: numpy.ndarray,
                             agent_action: numpy.ndarray,
                             reward: numpy.ndarray,
                             episode_done: numpy.ndarray,
                             agent_observation_next: numpy.ndarray,
                             warmup_step: int, warmup_episode: int):
        pass

    def complete_step_train(self,
                            logger: logging.Logger,
                            session,
                            interface: Interface,
                            agent_observation_current: numpy.ndarray,
                            agent_action: numpy.ndarray,
                            reward: numpy.ndarray,
                            episode_done: numpy.ndarray,
                            agent_observation_next: numpy.ndarray,
                            train_step: int, train_episode: int):
        # Save the current step in the buffer together with the current value estimate and the log-likelihood
        # self._model.buffer.store(agent_observation_current.copy(), agent_action.copy(), reward.copy(), episode_done.copy(), self._current_value_estimate.copy(), self._current_log_likelihood.copy())
        pass

    def complete_step_inference(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                agent_observation_current: numpy.ndarray,
                                agent_action: numpy.ndarray,
                                reward: numpy.ndarray,
                                episode_done: numpy.ndarray,
                                agent_observation_next: numpy.ndarray,
                                inference_step: int, inference_episode: int):
        pass

    def complete_episode_warmup(self,
                                logger: logging.Logger,
                                session,
                                interface: Interface,
                                last_step_reward: numpy.ndarray,
                                episode_total_reward: numpy.ndarray,
                                warmup_step: int, warmup_episode: int):
        pass

    def complete_episode_train(self,
                               logger: logging.Logger,
                               session,
                               interface: Interface,
                               last_step_reward: numpy.ndarray,
                               episode_total_reward: numpy.ndarray,
                               train_step: int, train_episode: int):
        pass

    def complete_episode_inference(self,
                                   logger: logging.Logger,
                                   session,
                                   interface: Interface,
                                   last_step_reward: numpy.ndarray,
                                   episode_total_reward: numpy.ndarray,
                                   inference_step: int, inference_episode: int):
        # TODO: Controlla se devi salvare cose
        pass

    @property
    def saved_variables(self):
        return None

    @property
    def warmup_steps(self) -> int:
        return 0
