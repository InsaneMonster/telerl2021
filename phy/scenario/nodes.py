#!/usr/bin/env python3
# filename "nodes.py"

import numpy as np
import phy.common.common_dict as dic


class Node:
    """Node definition"""
    def __init__(self, typ, dire, coord, ant, gain, max_pow):
        """Constructor of the Node class.

        Parameters
        ----------
        typ : str in {user_types, bs_types},
            it represents the type of the node.
        dire : str in {user_directions, bs_directions},
            it defines the communication direction of the node.
        coord : ndarray 1 x 2
            it represents the x,y cartesian coordinates of the node.
        ant : int > 0
            number of antennas of the node.
        gain : float,
            it represents the antenna gain of the node.
        max_pow : float,
            It represents the max power available on transmission;
            if node is DL, this parameter is always None.
        """
        # Control on INPUT
        if coord.shape != (2,):
            raise ValueError(f'Illegal positioning: for Node, '
                             f'coord.shape must be (2,), '
                             f'instead it is {coord.shape}')
        elif ant < 1 or (not isinstance(ant, int)):
            raise ValueError('ant must be a single integer >= 1')
        elif not isinstance(gain, (float, int)):
            raise ValueError('gain [dB] must be a single number')
        elif not isinstance(max_pow, (float, int)):
            raise ValueError('max_pow [dBm] must be a float or integer')
        # Set attributes
        self.type = typ
        self.dir = dire
        self.coord = coord
        self.ant = ant
        self.G = gain
        self.max_pow = max_pow if self.dir != 'DL' else -np.inf
        self.useful = None
        self.id = None
        self.ord = None
        # Noise definition (FOR NOW 174 dBm is the must)
        self.noise = RxNoise() if self.dir != 'UL' else None

    def connected(self, key: str):
        """"Returns the list of all users of some kind connected with self.
        It could be deprecated in future release.

        Parameters
        __________
        key : str in {user_dir, user_types}

        """
        try:
            if key in dic.user_dir:
                return [x for x in self.useful if x.dir == key]
            elif key in dic.user_types:
                return [x for x in self.useful if x.type == key]
            else:
                return []
        except TypeError:
            if self.useful.type == key or self.useful.dir == key:
                return [self.useful]
            else:
                return []
        except AttributeError:
            return []

    # Operator definition
    def __lt__(self, other):
        if isinstance(other, (self.__class__, self.__class__.__bases__)):
            return True if self.id < other.id else False
        else:
            raise TypeError('< applied for element of different classes')

    def __repr__(self):
        return f'{self.id}-{self.type:3}({self.dir})'
        # f'coord=({self.coord[0]:+06.1f},{self.coord[1]:+06.1f}))')


class Base(Node):
    """Node definition"""
    def __init__(self, typ, dire, coord, ant=None, gain=None, max_pow=None,
                 si_coef=None, r_shape=None):
        """BaseStation class for simulation purpose.

        Parameters
        ----------
        typ : str in bs_types,
              it represents the type of the node.
        dire : str in bs_directions,
               it defines the communication direction of the node.
        coord : ndarray 1 x 2
                it represents the x,y cartesian coordinates of the node.
        ant : int > 0
              number of antennas of the node. Default value is 1.
        gain : float representing antenna gain of the node. If typ is BS,
               default value is 15, else is 2.15.
        max_pow : float,
                  It represents the max power available on transmission.
                  If node is DL, this parameter is always None.
        si_coef : float,
                  self interference cancellation coefficient (FD node only)
        r_shape : float,
                  ratio of the power contributions by LOS path to the
                  remaining multi-paths. (FD node only)
        """
        # Control on input
        if typ not in dic.bs_types:
            raise ValueError(f'Illegal bs type. Available are: {dic.bs_types}')
        elif dire not in dic.bs_dir:
            raise ValueError(f'Illegal bs direction. BaseStation allows: {dic.bs_dir}')
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 15
        if max_pow is None:
            max_pow = 43
        if si_coef is None:
            si_coef = -110  # [dB]
        if r_shape is None:
            r_shape = 6  # [dB]
        # Init parent class
        super(Base, self).__init__(typ, dire, coord, ant, gain, max_pow)
        # Set attributes
        self.si_coef = si_coef if self.dir == 'FD' else -np.inf
        self.r_shape = r_shape if self.dir == 'FD' else 0


class User(Node):
    """User definition"""
    # TODO: change the beamformers to a general form xj = Bj sj
    #  and Pj = Tr{BjBj^H}
    def __init__(self, typ, dire, coord,
                 ant=None, gain=None, max_pow=None,
                 traffic=None, QoS=None, weight=None):
        """User class for simulation purpose.

        Parameters
        ----------
        typ : str in user_types,
            it represents the type of the node.
        dire : str in user_directions,
            it defines the communication direction of the node.
        traffic : str in traffic_types,
                it represents the kind of traffic used by the node
        coord : ndarray 1 x 2,
                it represents the x,y cartesian coordinates of the node.
        ant : int > 0
            number of antennas of the node. Default value is 1.
        gain : float, [dBi]
            It represents the antenna gain of the node. Default value 2.15.
        max_pow : float, [dBm]
                It represents the max power available on transmission in dBm.
                Default value is 24 dBm; if node is DL, this parameter is
                always None.
        QoS : tuple,
            it represents the quality of service demanded by the node;
            if traffic is eMBB, QoS is (min_throughput,);
            if traffic is URLLC, QoS is (output_pkt_rate, max_latency,
            outage_probability).
        weight : float,
            represent the user specific weight for fairness purpose

        """
        # Control on INPUT
        if typ not in dic.user_types:
            raise ValueError(f'Illegal user type. Available are: {dic.user_types}')
        elif dire not in dic.user_dir:
            raise ValueError(f'Illegal user direction. Available are: {dic.user_dir}')
        elif traffic is None:
            traffic = 'eMBB'
        elif traffic not in dic.traffic_types:
            raise ValueError(f'Illegal traffic type. Available are: {dic.traffic_types}')
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 2.15
        if max_pow is None:
            max_pow = 24
        if traffic == 'URLLC':
            QoS = (1, 2, 1e-5) if QoS in (None, []) else QoS
            weight = 1  # imposing weight = 1 for URLLC communication
        elif traffic == 'eMBB':
            QoS = (4,) if QoS in (None, []) else QoS
            weight = 1 if weight is None else weight
        # Init parent class
        super(User, self).__init__(typ, dire, coord, ant, gain, max_pow)
        # User parameters
        self.traffic = traffic
        self.QoS = QoS
        self.weight = weight
        # Beamformers
        # self._C = None
        # self._B = None

    # @property
    # def B(self):
    #     if self.type == 'D2D' and self.dir == 'DL':
    #         return self.useful.B
    #     else:
    #         return self._B
    #
    # @B.setter
    # def B(self, value):
    #     self._B = value
    #
    # @property
    # def C(self):
    #     if self.type == 'D2D' and self.dir == 'DL':
    #         return self.useful.C
    #     else:
    #         return self._C
    #
    # @C.setter
    # def C(self, value):
    #     self._C = value

    # TODO: solve this function!!!!
    # def int_seen(self, power, chan):
    #     """Provide the covariance matrices of the interference seen by the node
    #     for each freq resources
    #
    #         Parameters
    #         __________
    #         power : numpy.ndarray
    #             array F x K of power allocated per subcarrier and user
    #         chan : numpy.ndarray
    #             array of channel gain created by build_channel method
    #
    #         Output
    #         ______
    #         numpy.ndarray
    #         tensor with shape F x self.ant x self.ant; in practice, each
    #         dimension contains the covariance matrix of the interference.
    #     """
    #     user = fun.get_users()
    #     sub = chan.shape[0]
    #     # All the transmitters which interfere with r(self) and their antennas
    #     att = np.array([(n.ord, n.ant) if n.dir == 'UL' else
    #                     (n.useful.ord, n.useful.ant) for n in user
    #                     if n not in (self, self.useful)], dtype=int)
    #     t_int = att[:, 0]
    #     at_int = att[:, 1]
    #     a_tilde = np.sum(at_int)
    #     # The receiver of self
    #     r = [self.useful.ord if self.dir == 'UL' else self.ord]
    #     ar = self.useful.ant if self.dir == 'UL' else self.ant
    #     # Noise seen
    #     noise = (self.useful.noise.linear if self.dir == 'UL'
    #              else self.noise.linear)
    #     # Channel gains between t_int to r
    #     h = np.squeeze(chan[np.ix_(range(sub), t_int, r)], axis=2)
    #     H = np.transpose(np.dstack([np.hstack(h[f]) for f in range(sub)]),
    #                      axes=[2, 0, 1])
    #     # Transmission power employed
    #     P = np.zeros((sub, a_tilde, a_tilde))
    #     P[:, np.arange(a_tilde), np.arange(a_tilde)] = \
    #         np.repeat(power[:, user != self], at_int, axis=1)
    #     return (np.conj(self.C.transpose(0, 2, 1)) @ H @ P
    #             @ np.conj(H.transpose(0, 2, 1)) @ self.C
    #             + noise * np.eye(ar))


class RxNoise:
    """Represent the noise value at the physical receiver"""
    def __init__(self, linear=None, dB=None, dBm=-140):
        if (linear is None) and (dB is None):
            self.dBm = dBm
            self.dB = dBm - 30
            self.linear = 10 ** (self.dB / 10)
        elif (linear is not None) and (dB is None):
            self.linear = linear
            if self.linear != 0:
                self.dB = 10 * np.log10(self.linear)
                self.dBm = 10 * np.log10(self.linear * 1e3)
            else:
                self.dB = -np.inf
                self.dBm = -np.inf
        else:
            self.dB = dB
            self.dBm = dB + 30
            self.linear = 10 ** (self.dB / 10)

    def __repr__(self):
        return (f'noise({self.linear:.3e}, '
                f'dB={self.dB:.1f}, dBm={self.dBm:.1f})')
