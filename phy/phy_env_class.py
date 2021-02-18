#!/usr/bin/env python3
# file: phy_env_class.py

# Import
import os
import datetime
import numpy as np
from phy.common.common_method import randint_sum_equal_to, randint_sum_equal_to2
from phy.scenario.cluster import Cluster
from phy.scenario.resources import Resources
import phy.resource_allocation.waterfilling as wf

available_algorithms = {'time_freq', 'random'}
available_rl_versions = {'bernoulli'}
available_classes = np.array([[0, 1], [0.2, 0.8], [0.5, 0.5], [0.8, 0.2], [0, 1]])
mode_classes = {0: None, 1: [0, 1], 2: [0.2, 0.8], 3: [0.5, 0.5], 4: [0.8, 0.2], 5: [1, 0]}
param_under_test = np.around(np.arange(0.1, 0.6, 0.1), 2).tolist()


def std_param(activation):
    """Simply store the parameter of simulation"""
    freqs = 12
    slots = 10
    minislots = 14
    tolerable_latency = np.array([7])
    outage_prob = np.array([1e-5])            # not used for now
    urllc_pkt = np.array([activation]) if activation is not None else None
    downlink_users = 10
    uplink_users = 0
    target_rate = [4] * downlink_users       # not used for now
    return (freqs, slots, minislots, tolerable_latency, outage_prob,
            urllc_pkt, downlink_users, uplink_users, target_rate)


class Phy:
    def __init__(self,
                 freqs: int,
                 slots: int,
                 minislots: int,
                 tolerable_latency: np.ndarray,
                 outage_prob: np.ndarray,
                 pkt_arrival: np.ndarray,
                 downlink_users: int,
                 uplink_users: int,
                 target_rate: list,
                 rl_ver: str,
                 vision_ahead: int = 0,
                 cw_tot_number: int = None,
                 cw_class_prob: list = None,
                 q_norm: float = 0,
                 ra_algorithm: str = 'random',
                 seed: int = None,
                 env_name: str = 'phy_render',
                 reward_weight: float = .5,
                 render: bool = True,
                 render_path: str = None,
                 random_queue: bool = False):
        """Constructor of class Phy.
        Some important stuff to know:
            - eMBB : enhanced Mobile BroadBand. In the following it is a short for eMBB users, i.e.
                users having eMBB traffic type. In practice, this kind of users must transmit lot
                of data, so their aim is maximize the troughput.
            - URLLC : Ultra Reliable Low-Latency Communication. In the following it is a short for
                URLLC users, i.e. users having URLLC traffic type. These users must satisfy a
                latency constraint, expressed by max_latency. This latency is expressed in
                terms of minislots.
        """
        # Control on input
        assert len(tolerable_latency) == len(outage_prob)
        if ra_algorithm not in available_algorithms:
            raise ValueError(f'Resource allocation algorithm requested is not implemented. '
                             f'Possible are: {available_algorithms}')
        if rl_ver not in available_rl_versions:
            raise ValueError(f'RL environment requested is not implemented. '
                             f'Possible are: {available_rl_versions}')
        # render data
        if render_path is None:
            self.render_directory = os.path.join(
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "render"),
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        else:
            self.render_directory: str = render_path
        if render:
            if not os.path.isdir(self.render_directory):
                try:
                    os.makedirs(self.render_directory)
                except FileExistsError:
                    pass
            self.render_file: str = os.path.join(self.render_directory, env_name)
        # Simulation attributes
        self.rng = np.random.RandomState(seed)
        self.ra_algorithm = ra_algorithm
        self.rl_ver = rl_ver                    # number of the environment
        self.vision_ahead = vision_ahead      # store the number of step seen ahead
        self.random_queue = random_queue
        # Cluster store all the parameters of a group of cells
        self.cluster = Cluster(Resources([(2e9, 0, freqs)], slots, minislots), rng=self.rng)
        # eMBB
        self.UU = uplink_users  # number of uplink users to be initialized
        self.DU = downlink_users  # number of downlink users to be initialized
        self.embb = None  # store the eMBB users
        self.target_rate = target_rate  # list of minimum rate for embb users
        self.embb_capacity = None  # capacity available of eMBB users
        self.embb_state = None  # store the state related to eMBB users
        self.ra = None  # resource allocation structure scheduled for eMBB
        self.cw_class_prob = cw_class_prob
        self.cw_class_prob_current = None
        self.cw_total_number = cw_tot_number  # total number of codeword placed in the environment
        self.cw_len = None  # length of each codeword
        self.cw_ref = None  # original codeword allocation for eMBB
        self.cw = None  # present codeword allocation for eMBB
        self.cw_class = None  # class of each codeword
        self.cw_user_number = None  # number of codeword for each user
        self.cw_fun = None  # function related to of losing a codeword
        # URLLC
        self.urllc = None  # store the urllc users
        self.urllc_queue = None  # infinite packet queue, list of lists (one for each user)
        self.q_len = None  # store the length of each queue
        self.q_norm = q_norm  # store the normalization factor used for  q in the reward
        self.urllc_latency_left = None  # store the remaining latency available for each urllc (l_max-l_old)
        self.urllc_state = None  # store the state related to urllc users
        self.pkt_arrival = pkt_arrival  # store the packets arrival parameter
        self.pkt_arrival_current = None  # store the parameter of the the current episode
        self.tolerable_latency = tolerable_latency  # max tolerable latency in terms of minislot
        self.tolerable_outage = outage_prob  # min tolerable outage prob (NOT USED NOW)
        # RL attributes
        self.state_previous = None  # store the previous state
        self.reward = None  # reward of each step (for debug purpose)
        self.reward_weight = reward_weight  # reward weight for the return (not used now)
        self.episode_done = False  # flag stating if the episode is done
        self.step_number: int = 0  # actual step number
        self.step_max: int = minislots * slots - 1  # maximum number of steps
        self.episode_number: int = 0  # number of the episode
        if self.rl_ver == 'bernoulli':
            self.action_space = np.arange(self.cluster.subs + 1)
        else:
            raise ValueError('Version other than bernoulli are not implemented yet')
        self.action: np.array = np.array([])  # action chosen
        # Metric values
        self._embb_outage_counter = 0  # store the counter of embb codeword in outage
        self._urllc_delay_counter = 0  # store the counter of how many time the latency is violated
        self._residual_urllc_pkt = 0  # store the number of packets in the queue at the end of the episode

    def env_init(self):
        self.init_cell(radius=500, bs_max_pow=43)

    def env_reset(self):
        # Control urllc_prob. In case the probability is not defined a priori, at the beginning of
        # the episode it is randomly chosen
        if self.pkt_arrival is None:
            self.pkt_arrival_current = [self.rng.choice(param_under_test)]
        else:
            self.pkt_arrival_current = self.pkt_arrival
        if self.cw_class_prob is None:  # store the probability of a codeword being in a determined class
            self.cw_class_prob_current = available_classes[self.rng.choice(len(available_classes))]
        else:
            self.cw_class_prob_current = self.cw_class_prob
        # Reset environment parameters
        # Reset reward
        self.reward = np.zeros(self.step_max + 1)
        # Reset the queues
        self.urllc_queue = [[] for _ in range(len(self.tolerable_latency))]
        if self.random_queue:
            self.q_len = self.rng.choice(np.arange(self.tolerable_latency[0] - 1)) * np.ones(len(self.tolerable_latency), dtype=int)
            for i in range(len(self.tolerable_latency)):
                for j in range(self.q_len[i]):
                    self.urllc_queue[i] += [-(self.q_len[i] - j)]
        else:
            self.q_len = np.zeros(len(self.tolerable_latency))
        # Reset the latency
        self.urllc_latency_left = np.zeros(len(self.tolerable_latency), dtype=int)
        # Reset users
        self.reset_users(self.ra_algorithm)
        # Baseline eMBB resource and codeword allocation
        self.ra, self.cw_user_number, self.cw_len, self.cw, self.cw_fun = self.perform_ra(self.ra_algorithm)
        self.cw_ref = np.copy(self.cw)
        # Compute the capacity
        self.embb_capacity = np.sum(self.ra[0], axis=(0, 1)) / self.cluster.RB.minislot
        # RL param
        self.urllc_state = np.zeros(2)
        self.embb_state = np.zeros((1 + self.vision_ahead) * self.cluster.subs)
        self.state_previous = None
        self.episode_done = False
        self.step_number = -1
        self.episode_number += 1
        self.action = np.array([0])
        return self.env_get_state()

    def env_get_state(self, only_shape=False):
        # The following is needed to obtain the shape of the state
        if only_shape:
            if self.rl_ver == 'bernoulli':
                return (2 + (1 + self.vision_ahead) * self.cluster.subs,)
        self.state_previous = np.concatenate((self.urllc_state, self.embb_state))
        # eMMB state
        if self.rl_ver == 'bernoulli':
            # Future_sight gives the maximum step the agent is able to see in the future. Then, if the future is outside
            # the episode, the vector of embb_state is filled with zeros. Note that the +1 is needed because of the
            # python indexing that exclude always the last index in a slice. Note also that the number of zeros is
            # always a multiple of the frequency resources available.
            future_sight = self.step_number + 2 + self.vision_ahead
            if future_sight <= self.step_max + 1:
                self.embb_state = self.cw_fun[self.cw_ref[self.step_number + 1:future_sight]].flatten()
            elif future_sight > self.step_max + 1:
                self.embb_state = np.concatenate((self.cw_fun[self.cw_ref[self.step_number + 1:future_sight]].flatten(),
                                                  np.zeros((future_sight - self.step_max - 1) * self.cluster.subs)))
        # Control if a new URLLC packet is arrived only if the episode is not considered done
        if not self.episode_done:
            if self.rl_ver == 'bernoulli':
                new_pkt = self.rng.binomial(1, self.pkt_arrival_current)
            elif self.rl_ver == 'poisson':
                new_pkt = self.rng.poisson(self.pkt_arrival_current)
            else:
                new_pkt = [1]
            # Update urllc state
            for i, pkt in enumerate(new_pkt):  # i is the URLL user
                if pkt:
                    self.urllc_queue[i] += [self.step_number] * new_pkt[i]
                self.urllc_latency_left[i] = self.tolerable_latency - (self.step_number + 1 - self.urllc_queue[i][0]
                                                                       if len(self.urllc_queue[i]) >= 1 else 0)
                self.q_len[i] = len(self.urllc_queue[i])
        self.urllc_state = np.concatenate((self.q_len, self.urllc_latency_left))
        # Merging data
        self._residual_urllc_pkt = self.q_len
        return np.concatenate((self.urllc_state, self.embb_state))

    def env_step(self, action):
        # UPDATE STEP NUMBER
        self.step_number += 1
        pun = self.action_space[action]
        outage = 0  # initialization: no codeword in outage because of my action
        # ACTION: Puncturing mode
        if pun:  # If pun == 0 the action means not transmit the URLLC pkt
            # Eliminate the oldest packet from the queue
            self.urllc_queue[0].pop(0)
            # Reduce the length of the Q
            self.q_len -= 1
            # Find the frequency to be punctured
            freq_used = pun - 1
            # Puncturing the time-freq resource
            self.ra[0][self.step_number, freq_used] = 0
            self.ra[1][self.step_number, freq_used] = 0
            self.ra[2][self.step_number, freq_used] = 0
            # Update the eMBB user capacity estimated
            self.embb_capacity = np.sum(self.ra[0], axis=(0, 1)) / self.cluster.RB.minislot
            # Puncturing the specific codeword
            self.cw[self.step_number, freq_used] = -1
            # Compute the terms needed for the reward
            if self.rl_ver == 'bernoulli':
                # Compute cw_fun
                if self.cw_fun[self.cw_ref[self.step_number, freq_used]] >= 0:
                    self.cw_fun[self.cw_ref[self.step_number, freq_used]] -= 1
                # If the punctured codeword was NOT in outage and NOW is in outage
                if (self.embb_state[freq_used] >= 0) and (self.cw_fun[self.cw_ref[self.step_number, freq_used]] == -1):
                    outage = -1  # codeword is in outage because of my action
        # Save the action
        self.action = np.array([pun])
        # Latency control
        if self.rl_ver == 'bernoulli':
            latency_fun = (self.urllc_latency_left[0] >= 0) * 0 + (self.urllc_latency_left[0] < 0) * 3 * (self.step_max + 1) / (self.cluster.subs + 1)
        else:
            latency_fun = 0
        # REWARD
        # Reward must be a function of the product of the correct probability.
        # Moreover, the episode should be considered finished if urllc constraint is violate
        if self.rl_ver == 'bernoulli':
            self.reward[self.step_number] = outage - latency_fun - self.q_norm * self.q_len
        # Control if the latency is violated or the episode is finished
        if np.any(self.urllc_latency_left < 0):
            self._embb_outage_counter = np.count_nonzero(self.cw_fun == -1)
            self._urllc_delay_counter = 1
            self.episode_done = True
        elif self.step_number == self.step_max:
            self._embb_outage_counter = np.count_nonzero(self.cw_fun == -1)
            self._urllc_delay_counter = 0
            self.episode_done = True
        return self.env_get_state(), self.reward[self.step_number], self.episode_done

    def env_sample_action(self):
        if len(self.urllc_queue[0]) > 0:
            return self.rng.choice(self.action_space)
        else:
            return 0

    def env_render(self):
        _render_file = self.render_file + "_ep_" + str(self.episode_number) + ".txt"
        if self.step_number == 0:
            try:
                os.remove(_render_file)
            except OSError:
                pass
        if self.step_number <= self.step_max:
            with open(_render_file, "a") as rf:
                data_str = '_____________________________________________________\n'
                data_str += str(self.step_number) + ','  # STEP
                data_str += str(self.action[0]) + ','  # ACTIONS
                data_str += f'{self.reward[self.step_number]:.4f},\n'  # REWARD
                data_str += ','.join(f'{el:.0f}' for el in self.state_previous) + '\n'
                data_str += ','.join(f'{el:.0f}' for el in np.concatenate((self.urllc_state, self.embb_state)))  # STATE
                if self.rl_ver <= 4:
                    data_str += ',' + str(self.pkt_arrival_current)  # URLLC PROB
                data_str += "\n"
                data_str += '_____________________________________________________\n'
                rf.write(data_str)

    # Useful methods
    def init_cell(self,
                  radius: int = 500,
                  bs_max_pow: float = 43):
        # Create cell
        self.cluster.wipe_cells()
        self.cluster.place_cell(1, r_outer=radius, r_inner=30)
        self.cluster.cell[0].place_bs(1, 'mBS', 'HD', max_pow=bs_max_pow)

    def reset_users(self, algorithm: str):
        # Eliminate old users from the cell
        self.cluster.wipe_users()
        # Insert users
        tr = [(r,) for r in self.target_rate]
        for c in self.cluster.cell:
            c.place_user(self.DU, 'UE', 'DL', traffic='eMBB', QoS=tr[:self.DU], ant=1)
            c.place_user(self.UU, 'UE', 'UL', traffic='eMBB', QoS=tr[self.DU:], ant=1)
            for i in range(len(self.tolerable_latency)):
                c.place_user(1, 'UE', 'DL', traffic='URLLC', QoS=(self.pkt_arrival_current[i],
                                                                  self.tolerable_latency[i],
                                                                  self.tolerable_outage[i]))
        if algorithm == 'time_freq':
            # Build gain channel tensor
            self.cluster.build_chan_gain()
        # Environment parameters
        self.embb = self.cluster.get_user('eMBB')
        self.urllc = self.cluster.get_user('URLLC')

    def perform_ra(self, algorithm: str):
        """Return the resource allocation grid and the codeword placement grid.
            TODO: create a class for codeword and a class for resource allocation grid
        """
        # Name definitions
        slots = self.cluster.RB.slot
        minislots = self.cluster.RB.minislot
        frequencies = self.cluster.subs
        users = len(self.embb)
        # Resource Allocation
        if algorithm == 'random':
            ra = [np.zeros((slots, frequencies, users)),
                  np.zeros((slots, frequencies, users)),
                  np.zeros((slots, frequencies, users), dtype=bool)]
            # Generate a random allocation of users on the grid: a[t, f] give the user (number)
            # allocated on time slot t and frequency f
            a = self.rng.randint(users, size=(slots, frequencies))
            # Provide the allocation in freq-time matrix
            for n in np.arange(users):
                ra[0][np.nonzero(a == n)[0], np.nonzero(a == n)[1], n] = 1
                ra[1][np.nonzero(a == n)[0], np.nonzero(a == n)[1], n] = 1
                ra[2][np.nonzero(a == n)[0], np.nonzero(a == n)[1], n] = 1
        elif algorithm == 'time_freq':
            ra = wf.time_freq_ra(self.cluster.chan_gain, self.embb, self.cluster.RB)
        else:
            raise ValueError('Algorithm not implemented.')
        # Evaluate the number of frequencies allocated to each user
        # freq_per_user is T x K, hence the FR taken by user k in the whole coherence interval is freq_per_user[:, k]
        freq_per_user = np.sum(ra[2], axis=1)
        # Mini slot allocation
        for i in range(len(ra)):
            ra[i] = np.repeat(ra[i], minislots, axis=0)
        # Compute the number of codeword per user:
        if self.cw_total_number is None:
            # If the length of each codeword is equal to M (hence a slot), cw_number is equal to the number of FR
            # allocated to each user
            cw_number = np.sum(freq_per_user, axis=0)
            # The length is then set as
            cw_len = minislots * np.ones(np.sum(cw_number), dtype=int)
        else:
            # Condition of convergence
            # min_cw_len <= F * M * T / self.cw_total_number
            # lower <= self.cw_total_number / K
            # cw_number >= np.sum(freq_per_user, axis=0) * M / min_cw_len
            min_cw_len = minislots // 3
            cw_number = randint_sum_equal_to(self.cw_total_number, n=users, lower=1,
                                             upper=np.floor(np.sum(freq_per_user, axis=0) * minislots / min_cw_len))
            # Compute the length of each codeword
            cw_len = np.zeros(np.sum(cw_number), dtype=int)
            try:
                for k in range(users):
                    # Select the codeword of each user and obtain the length of each codeword considering that the
                    # sum of all the codewords of user k must be equal to the resources given to k in terms of
                    # minislots, i.e. sum(freq_per_user[:, k])*M
                    cw_len[sum(cw_number[:k]):sum(cw_number[:k + 1])] = randint_sum_equal_to2(sum(freq_per_user[:, k]) * minislots,
                                                                                              n=cw_number[k], lower=min_cw_len)
            except ValueError:
                # In rare case of error, let's organize all the codeword as fixed length
                cw_number = np.sum(freq_per_user, axis=0)
                # The length is then set as
                cw_len = minislots * np.ones(np.sum(cw_number), dtype=int)
                # Place the codewords in the allocated resources
        cw = np.empty((slots * minislots, frequencies), dtype=int)
        for k in range(users):
            cw[ra[2][:, :, k]] = np.repeat(np.arange(sum(cw_number[:k]), sum(cw_number[:k + 1])),
                                           cw_len[sum(cw_number[:k]):sum(cw_number[:k + 1])])
        # Compute the class of codeword and the reward function associated of each codeword
        cw_fun = np.zeros(np.sum(cw_number))
        if self.rl_ver == 'bernoulli':
            self.cw_class = self.choose_class(sum(cw_number), self.cw_class_prob_current)
            cw_fun = self.cw_class
        return ra, cw_number, cw_len, cw, cw_fun

    def choose_class(self, n: int, class_percent: list) -> np.ndarray:
        """ The function returns the correct number of codewords of different classes
        Parameters
        ___________
        n: int,
            number of codewords to be returned.
        class_percent : list
            a list containing the percentage of each class; the index represent the class of the codeword.
        """
        # Normalization of percentage list
        percent = np.array(class_percent) / sum(class_percent)
        # Evaluate the codeword class
        return np.nonzero(self.rng.multinomial(1, percent, size=n))[1]

    @property
    def embb_outage_counter(self):
        return self._embb_outage_counter

    @property
    def urllc_delay_counter(self):
        return self._urllc_delay_counter

    @property
    def residual_urllc_pkt(self):
        return self._residual_urllc_pkt

    @property
    def urllc_state_range(self):
        return np.arange(self.urllc_state.shape[0])

    @property
    def embb_state_range(self):
        return np.arange(self.urllc_state.shape[0], self.embb_state.shape[0] + self.urllc_state.shape[0])

    @property
    def embb_state_range_present(self):
        return np.arange(self.cluster.subs) + self.urllc_state.shape[0]
