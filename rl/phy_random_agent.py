# Import packages

import logging
import numpy
import tensorflow

# Import usienarl

from usienarl import Agent, Interface, SpaceType


class RandomAgent(Agent):
    """
    Random agents taking random actions in all modes.
    It does not require warm-up.

    Note: no save/restore is ever performed by this agent.
    """

    def __init__(self,
                 name: str):
        # Generate base agent
        super(RandomAgent, self).__init__(name)

    def setup(self,
              logger: logging.Logger,
              scope: str,
              parallel: int,
              observation_space_type: SpaceType, observation_space_shape: (),
              agent_action_space_type: SpaceType, agent_action_space_shape: (),
              summary_path: str = None, save_path: str = None, saves_to_keep: int = 0) -> bool:
        # Make sure parameters are correct
        assert(parallel > 0)
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
        # Act randomly
        action = []
        for i in range(self.parallel):
            if agent_observation_current[i, 0] == 0:
                action.append(0)
            else:
                if interface.environment.rng(i).choice([0, 1]):
                    possible_actions = [i for i in interface.environment.possible_actions(logger, session)[i] if i != 0]
                    action.append(interface.environment.rng(i).choice(possible_actions))
                else:
                    action.append(0)
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
        pass

    @property
    def saved_variables(self):
        return None

    @property
    def warmup_steps(self) -> int:
        return 0
