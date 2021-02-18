#!/usr/bin/env python3
# filename "cells.py"

import numpy as np
import phy.common.common_dict as dic
from phy.scenario.nodes import User, Base
import phy.common.common_method as common


# Cell Classes
class Cell:
    """Class Cell creates a macro-cell with specific parameters
    and node. Calling Cell() will create a standard macro-cell
    with a single half-duplex BS in its center."""
    def __init__(self, r_outer: float, r_inner: float,
                 coord: np.ndarray, pl_exp: float, sh_std: float, nid: int = 0,
                 rng: np.random.RandomState = None):
        """Constructor of the cell.

        Parameters
        ----------
        r_outer : float,
            outer radius (nodes can be placed between outer and inner).
        r_inner : float,
            inner radius (nodes can be placed between outer and inner).
        coord : np.ndarray with shape (2,),
            it is the absolute positioning of the cell, in term of x,y
            cartesian coordinates of the cell. Node positioning is relative
            to the coordinates of the cell.
        pl_exp : float,
            path loss exponent in the cell.
        sh_std : float,
            standard deviation of the shadowing phenomena.
        nid : int,
            number representing the id of the cell.
        """
        if (r_outer < r_inner) or (r_inner <= 0) or (r_outer <= 0):
            raise ValueError('r_outer and r_inner must be >= 0 and ' +
                             'r_outer > r_inner')
        elif coord.shape != (2,):
            raise ValueError(f'Illegal positioning: coord.shape must be (2,)')
        elif pl_exp <= 0:
            raise ValueError('pl_exp must be >= 0')
        else:
            # Physical attributes
            self.coord = coord
            self.r_outer = r_outer
            self.r_inner = r_inner
            self.pl_exp = pl_exp
            self.sh_std = sh_std
            self.id = nid
            # Random State generator
            self.rng = np.random.RandomState() if rng is None else rng
            # List of cells data
            self.prev = 0       # store the number of nodes of all previous instantiated cells
            self.next = None    # a pointer to the next instantiated cell
            # Macro-BS
            self.node = []
            self.place_bs(1, 'mBS', 'HD', np.array([[0, 0]]), 1)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return True if self.id < other.id else False
        else:
            raise TypeError('< applied for element of different classes')

    def place_bs(self, n: int, typ: str,
                 d=None, coord=None, ant=None, gain=None, max_pow=None,
                 si_coef=None, r_shape=None):
        """Place a predefined number n of bs in the cell.

        Parameters
        ----------
        n : int > 0 representing the number of nodes to be placed.
            If mBS node N must be exactly 1.
        typ : str in node_types, representing the type of node to be placed.
        d : is str in bs_directions and represents the duplex technology to be
            considered.
        coord : N x 2 ndarray; row i represents the x, y cartesian
                coordinates of node i.
        ant : list of int > 0 representing the number of antennas of each
              bs; if a single int, each bs will have same number of antennas.
        gain : sequence of int or float, representing the antenna gain used in
               the path loss computation; if a single int, each bs will have
               same gain values.
        max_pow : list of int or float, representing the maximum power
                  available on the bs; if a single int, each bs will have
                  same max_pow.
        si_coef : list of int or float, representing the self-interference
                  cancellation factor (FD mode); if a single number, each bs
                  will have same si_coef.
        r_shape : list of int or float, representing the ratio of the power
                  contributions by LOS path to the remaining multi-paths for SI
                  (FD node only); if a single number, each bs will have same
                  r_shape.
        """
        # Control on INPUT
        assert typ in dic.bs_types
        # sub = fun.get_sub()
        # sub = sum([x.N_sc for x in ResourceBlock.__ref__[ResourceBlock]])
        # if sub == 0:
        #     raise TypeError(f'No subcarrier available. Please instantiate '
        #                     f'a Resources element before the creation '
        #                     f'of the environment')
        if not isinstance(n, int) or (n < 0):
            raise ValueError('N must be int >= 0')
        elif n == 0:
            return
        # Input reformat
        if not isinstance(ant, list):
            ant = [ant] * n
        if not isinstance(gain, list):
            gain = [gain] * n
        if not isinstance(max_pow, list):
            max_pow = [max_pow] * n
        if not isinstance(si_coef, list):
            si_coef = [si_coef] * n
        if not isinstance(r_shape, list):
            r_shape = [r_shape] * n
        # Counting the actual nodes
        n_old = len(self.node)
        # In case of mBS
        if typ == "mBS":
            if n > 1:
                raise ValueError('Only 1 mBS per cell')
            else:
                dire = [d]
                try:
                    self.node.pop(0)
                    # warnings.warn('mBS already set, new mBS will take place')
                except IndexError:
                    pass
                coord = np.zeros((n, 2)) if coord is None else coord
        # In case of fBS
        elif typ == 'fBS':
            dire = [d] * n
            if coord is None:
                rho, phi = common.circ_uniform(n, self.r_outer, self.r_inner, self.rng)
                coord = np.hstack((rho * np.cos(phi), rho * np.sin(phi)))
        # Append nodes
        for i in range(n):
            self.node.append(Base(typ=typ, dire=dire[i],
                                  coord=coord[i], ant=ant[i],
                                  gain=gain[i], max_pow=max_pow[i],
                                  si_coef=si_coef[i], r_shape=r_shape[i]))
        # Set intended channel and beamforming
        # If the BS is changed, all the UE must refer to the new BS
        if typ == "mBS":
            self.node[-1].useful = []
            for x in self.node:
                if x.type == 'UE':
                    self.node[-1].useful.append(x)
                    x.useful = self.node[-1]
                    # if x.dir == 'DL':
                    #     x.C = np.repeat(np.eye(x.ant)
                    #                     [np.newaxis, :, :], sub, axis=0)
                    #     x.B = np.repeat(np.eye(x.useful.ant, x.ant)
                    #                     [np.newaxis, :, :], sub, axis=0)
                    # else:
                    #     x.C = np.repeat(np.eye(x.useful.ant)
                    #                     [np.newaxis, :, :], sub, axis=0)
                    #     x.B = np.repeat(np.eye(x.ant, x.useful.ant)
                    #                     [np.newaxis, :, :], sub, axis=0)
        # TODO: femto BS communication is not implemented
        elif typ == "fBS":
            for i in range(n_old, n_old + n):
                self.node[i].useful = self.node[0]
        # Order nodes
        self.order_nodes()
        self.update_id()

    def wipe_bases(self):
        """This method wipe out the bases in the node list"""
        users = []
        for n in self.node:
            if isinstance(n, User):
                users.append(n)
                if n.type == 'UE':
                    n.useful = None
        self.node = users
        self.order_nodes()
        self.update_id()

    def place_user(self, n: int, typ: str,
                   d=None, coord=None, ant=None, gain=None, max_pow=None,
                   traffic=None, QoS=None, weight=None):
        """Place a predefined number n of nodes in the cell.

        Parameters
        ----------
        n : int > 0, representing the number of user to be placed.
            If typ = "D2D", N represents the number of D2D couples, so the
            number of user will be 2N
        typ : str in user_types, representing the type of user to be placed.
        d : if typ is not "D2D", d is str in user_directions and represents the
            direction of nodes to be placed; if typ is "D2D", d is the maximum
            distances between transmitter and receiver in the D2D couples.
        coord : N x 2 ndarray; row i represents the x, y cartesian
                coordinates of node i.
        ant : list of int > 0 representing the number of antennas of each
                node; if a single int, each user will have same number of
                antennas.
        gain : list of int or float, representing the antenna gain used in
               the path loss computation; if a single int, each user will have
               same gain values.
        max_pow : list of int or float, representing the maximum power
                  available on the node; if a single int, each node will have
                  same max_pow.
        traffic : list of str, representing the kind of traffic processed
                  by the user; if a single str, each user will have same
                  traffic.
        QoS : list of tuple, representing the quality of service required
              by the users; if a single tuple, each user will have same QoS.
        weight : list of float, with 0 < float <= 1, representing the
                importance of the node in the scheduler process;  if a single
                weight, each user will have same weight.
        """
        # Control on INPUT
        assert typ in dic.user_types, f'typ must be in {dic.user_types}'
        if not isinstance(n, int) or (n < 0):   # Cannot add a negative number od nodes
            raise ValueError('N must be int >= 0')
        elif n == 0:    # No node to be added
            return

        # Counting the present nodes
        n_old = len(self.node)
        # In case of UE
        if typ == 'UE':
            dire = [d] * n
            if coord is None:
                rho, phi = common.circ_uniform(n, self.r_outer, self.r_inner, self.rng)
                coord = np.hstack((rho * np.cos(phi), rho * np.sin(phi)))
        # In case of D2D couple
        elif typ == 'D2D':
            d2d_max = 50 if d is None else d
            if (not isinstance(d2d_max, (int, float))) or (d2d_max < 1):
                raise ValueError('For D2D couples, d represent the maximum ' +
                                 'distance [m] between transmitter and ' +
                                 'receiver. Hence, it must be a number > 1')
            d2d_min = 0
            dire = ['UL'] * n + ['DL'] * n
            if coord is None:
                rho, phi = common.circ_uniform(n, self.r_outer, self.r_inner, self.rng)
                p_tx = rho * np.hstack((np.cos(phi), np.sin(phi)))
                rho, phi = common.circ_uniform(n, d2d_max, d2d_min, self.rng)
                p_rx = p_tx + (rho * np.hstack((np.cos(phi), np.sin(phi))))
                coord = np.vstack((p_tx, p_rx))
                # Test if anyone goes outside the borders
                try:
                    rho = np.linalg.norm(coord, axis=1)
                    phi = np.arctan2(coord[:, 1], coord[:, 0])
                    coord[rho > self.r_outer, :] = \
                        self.r_outer * np.hstack((np.cos(phi[rho > self.r_outer]),
                                                  np.sin(phi[rho > self.r_outer])))
                except ValueError:
                    pass
            n = 2 * n
        # Input reformat
        if not isinstance(ant, list):
            ant = [ant] * n
        if not isinstance(gain, list):
            gain = [gain] * n
        if not isinstance(max_pow, list):
            max_pow = [max_pow] * n
        if not isinstance(QoS, list):
            QoS = [QoS] * n
        if not isinstance(weight, list):
            weight = [weight] * n
        # Append nodes
        for i in range(n):
            self.node.append(User(typ=typ, dire=dire[i],
                                  coord=coord[i], ant=ant[i],
                                  gain=gain[i], max_pow=max_pow[i],
                                  traffic=traffic, QoS=QoS[i], weight=weight[i]))
        # Set intended channel and beamforming
        # If UE nodes are added, they have to refer to the BS
        if typ == "UE":
            for i in range(n_old, n_old + n):
                self.node[0].useful.append(self.node[i])
                self.node[i].useful = self.node[0]
                # if self.node[i].dir == 'DL':
                #     self.node[i].C = np.repeat(np.eye(self.node[i].ant)
                #                                [np.newaxis, :, :], sub, axis=0)
                #     self.node[i].B = np.repeat(
                #         np.eye(self.node[i].useful.ant, self.node[i].ant)
                #         [np.newaxis, :, :], sub, axis=0)
                # else:
                #     self.node[i].C = np.repeat(np.eye(self.node[i].useful.ant)
                #                                [np.newaxis, :, :], sub, axis=0)
                #     self.node[i].B = np.repeat(
                #         np.eye(self.node[i].ant, self.node[i].useful.ant)
                #         [np.newaxis, :, :], sub, axis=0)
        # If D2D node are added, tx and rx have to refer to each other
        elif typ == "D2D":
            for i in range(n_old, n_old + n // 2):
                self.node[i].useful = self.node[i + n // 2]
                self.node[i + n // 2].useful = self.node[i]
                # if self.node[i].dir == 'DL':
                #     self.node[i].C = np.repeat(np.eye(self.node[i].ant)
                #                                [np.newaxis, :, :], sub, axis=0)
                #     self.node[i].B = np.repeat(
                #         np.eye(self.node[i].useful.ant, self.node[i].ant)
                #         [np.newaxis, :, :], sub, axis=0)
                # else:
                #     self.node[i].C = np.repeat(np.eye(self.node[i].useful.ant)
                #                                [np.newaxis, :, :], sub, axis=0)
                #     self.node[i].B = np.repeat(
                #         np.eye(self.node[i].ant, self.node[i].useful.ant)
                #         [np.newaxis, :, :], sub, axis=0)
        # Order nodes
        self.order_nodes()
        self.update_id()

    def wipe_users(self):
        """This method wipe out the users in the node list"""
        bs = []
        for n in self.node:
            if isinstance(n, Base):
                n.useful = [i for i in n.useful if isinstance(i, Base)]
                bs.append(n)
        self.node = bs
        self.order_nodes()
        self.update_id()

    def order_nodes(self):
        """The method orders the nodes following the order given by type:
            'mBS', 'fBS', 'UE', 'D2D'; and direction: 'HD', 'FD', 'UL','DL'
        """
        self.node.sort(key=lambda x: (dic.node_types[x.type], dic.node_dir[x.dir]))

    def update_id(self):
        """ The method updates the id of each node in the cell.
        It propagates the updating to the next cell instantiated.

        Returns
        ______
        node.id : tuple,
            the id of the node which is formed by the id of the cell and
            the number of the node in the cell, following the order given
            by order_nodes.
        node.ord : int,
            an ordered counter of the node built according to the channel
            gain tensor order.
        """
        for i, node in enumerate(self.node):
            node.id = (self.id, i)
            node.ord = self.prev + i
        if self.next is not None:
            self.next.prev = self.prev + len(self.node)
            self.next.update_id()

    # Indexing methods
    def ind(self):
        """The function returns the index of each node in the cell. It is not
        efficient and not used frequently.
        """
        i_typ = [None] * len(dic.node_types)
        i_dir = [None] * len(dic.node_dir)
        for j in range(len(dic.node_types)):
            i_typ[j] = [i for i, x in enumerate(self.node)
                        if x.type == list(dic.node_types)[j]]
        for j in range(len(dic.node_dir)):
            i_dir[j] = [i for i, x in enumerate(self.node)
                        if x.dir == list(dic.node_dir)[j]]
        return i_typ, i_dir

    def count_elem(self):
        count = [[0] * 3 for _ in range(len(dic.node_types))]
        ind = self.ind()[0]
        for j in range(len(dic.node_types)):
            count[j][0] = len(ind[j])
            if count[j][0]:
                ls = (list(dic.bs_dir) if j in range(len(dic.bs_dir))
                      else list(dic.user_dir))
                count[j][1] = [self.node[i].dir for i in ind[j]].count(ls[0])
                count[j][2] = [self.node[i].dir for i in ind[j]].count(ls[1])
        return count

    # Visualization methods
    def list_nodes(self, typ=None):
        if typ is None:
            ls = '\n'.join(f'{i:2} {n}' for i, n in enumerate(self.node))
        elif typ in dic.node_types:
            ls = '\n'.join(f'{i:2} {n}' for i, n in enumerate(self.node)
                           if n.type == typ)
        else:
            raise ValueError(f'Node type must be in {dic.node_types}')
        return print(ls)

    def __repr__(self):
        count = self.count_elem()
        line = []
        for i in range(len(dic.node_types)):
            ls = (list(dic.bs_dir) if i in range(len(dic.bs_dir))
                  else list(dic.user_dir))
            line.append(f'{list(dic.node_types)[i]:3} = {count[i][0]:02}, '
                        f'{ls[0]:2} = {count[i][1]:02}, '
                        f'{ls[1]:2} = {count[i][2]:02};')
        counter = '\n\t\t'.join(str(n) for n in line)
        return (f'cell #{self.id}: '
                f'\n\tdim:\t{self.r_inner}-{self.r_outer} [m]'
                f'\n\tcoord:\t{self.coord[0]:+04.0f},{self.coord[1]:+04.0f} [m]'
                f'\n\tcoef:\tPL_exp = {self.pl_exp:1},'
                f'\tSH = {self.sh_std:02} [dB]'
                f'\n\tnodes:\t{counter}\n')
