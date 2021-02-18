#!/usr/bin/env python3
# filename "cluster.py"
# This file contains the class cluster in which a cluster of cells with shared
# resources is created and managed. Furthermore, the class contains all the
# common method used to create the channel gain tensor and collect all the
# users in the cluster.

import matplotlib.pyplot as plt
import matplotlib.lines as mln
import numpy as np
import warnings
from collections import OrderedDict
from matplotlib.patches import Circle
from matplotlib import rc
from scipy.constants import c as light_speed
from tabulate import tabulate
# Import from phy
import phy.common.common_dict as dic
import phy.common.common_method as common
from phy.scenario.resources import Resources
from phy.scenario.cells import Cell


class Cluster:
    """Class containing the instances of cell and users for a single
    simulation"""
    def __init__(self, RB: Resources,
                 rng: np.random.RandomState = None,
                 frau: float = 10):
        """Constructor of the class.

        TODO: create a wrapper that bang the creation of the simulation environment in the right
            order (the most efficient one)

        Parameters
        __________
        RB  : Resources,
            instance of class resources giving frequency and time resources
            available in the whole cluster.
        seed : int,
            seed of the pseudorandom number generator for the simulation.
        frau : int,
            upper bound of the distance considered for the Fraunhofer region
        """
        # Parameters
        self.RB = RB            # store the resources available in the cluster
        self.cell = []          # store the cells considered in the cluster
        self.chan_gain = None   # store the channel gain tensor
        self.frau = frau        # store the distance assumed for the Fraunhofer
        # Random State Generator
        self.rng = np.random.RandomState() if rng is None else rng

    def place_cell(self, n: int,
                   r_outer: (float, list) = 100,
                   r_inner: (float, list) = 10,
                   coord: (np.ndarray, list) = np.zeros(2),
                   pl_exp: (float, list) = 4,
                   sh_std: (float, list) = 6):
        """Place the cell in the cluster. If the arguments are a single element
        the same argument is used for all the instantiated cell.

        TODO:
            - impose that different cells have different positions

        Parameters
        __________
        n   : int
            number of cell to be initialized
        r_outer : (float, list),
            it represents the outer radius of the cells to be instantiated
        coord : (np.ndarray, list),
            it is the absolute positioning of the cell, in term of x,y
            cartesian coordinates of the cell.
        pl_exp : (float, list),
            path loss exponent in the cell.
        sh_std :  (float, list),
            standard deviation of the shadowing phenomena in the cell.
        """
        # Control on input
        if not isinstance(n, int) or n < 1:
            raise TypeError('n must be an int > 0')
        # Take the number of cell already instantiated
        n_old = len(self.cell)
        # Input reformat
        if not isinstance(r_outer, (tuple, list)):
            r_outer = [r_outer] * n
        if not isinstance(r_inner, (tuple, list)):
            r_inner = [r_inner] * n
        if not isinstance(coord, (tuple, list)):
            coord = [coord] * n
        if not isinstance(pl_exp, (tuple, list)):
            pl_exp = [pl_exp] * n
        if not isinstance(sh_std, (tuple, list)):
            sh_std = [sh_std] * n
        for i in range(n):
            self.cell.append(Cell(r_outer[i], r_inner[i], coord[i], pl_exp[i],
                                  sh_std[i], n_old + i, rng=self.rng))
            if n_old + i - 1 >= 0:
                self.cell[n_old + i - 1].next = self.cell[n_old + i]
            # if n_old + i -1 < 0 the last element of the list would point to the first creating
            # an infinite loop

    def wipe_cells(self):
        self.cell = []

    def wipe_users(self):
        for c in reversed(self.cell):
            bs = []
            for n in c.node:
                if n.type in dic.bs_types:
                    n.useful = [i for i in n.useful if i.type in dic.bs_types]
                    bs.append(n)
            c.node = bs
            c.order_nodes()
            if c.id == 0:
                c.update_id()

    @property
    def subs(self):
        return sum([x.N_sc for x in self.RB])

    @property
    def nodes(self):
        return [n for c in self.cell for n in c.node]

    # Get parameters methods
    def get_user(self, partition: str = None) -> []:
        """The function output a list of users in the cluster."""
        if partition in dic.traffic_types:
            return [n for n in self.nodes
                    if (n.type == 'UE' or (n.type == 'D2D' and n.dir == 'UL'))
                    and n.traffic == partition]
        elif partition in dic.user_types:
            return [n for n in self.nodes if n.type == partition]
        elif partition in dic.user_dir:
            return [n for n in self.nodes
                    if n.type in dic.user_types and n.dir == partition]
        else:
            return [n for n in self.nodes
                    if n.type == 'UE' or (n.type == 'D2D' and n.dir == 'UL')]

    def get_bs(self, partition: str = None) -> []:
        """The function output a list of bs in the cluster."""
        if partition in dic.bs_types:
            return [n for n in self.nodes if n.type == partition]
        elif partition in dic.user_dir:
            return [n for n in self.nodes
                    if n.type in dic.bs_types and n.dir == partition]
        else:
            return [n for n in self.nodes
                    if n.type == 'UE' or (n.type == 'D2D' and n.dir == 'UL')]

    # Channel build
    def build_chan_gain(self):
        """Create the list of all possible channel gain in the simulation.
        TODO:
            - squeeze matrix if n.ant = 1 everywhere
            - rebuild the version where the tensor is f x K x K x max(ant) x
                max(ant), change the method involved accordingly

        Returns
        -------
        h : ndarray[f, j, i][k, l]
            contains all the channel gains, where:
            [k, l] is the element of the matrix Nr x Nt for the MIMO setting
            j is the transmitter
            i is the receiver
            f is the subcarrier
        """
        # collect data
        cells = self.cell
        nodes = self.nodes
        subs = self.subs
        d0 = self.frau
        # Data struct definition
        data = np.array([[[np.zeros((i.ant, j.ant), dtype=complex)
                           for i in nodes]
                          for j in nodes]
                         for _ in range(subs)])
        # Frequency of the subcarriers
        freq_sc = [r.f0 + f * r.d_f for r in self.RB for f in range(r.N_sc)]
        # For loop
        for f in range(subs):
            for i, n_i in enumerate(nodes):  # i is the receiver
                c_i = cells[n_i.id[0]]  # c_i is the cell of the receiver
                # Channel reciprocity
                for j, _ in enumerate(nodes[:i]):  # j is the transmitter
                    data[f, j, i] = data[f, i, j].T
                # Channel computation
                for j, n_j in enumerate(nodes[i:], i):  # j is the transmitter
                    c_j = cells[
                        n_j.id[0]]  # c_j is the cell of the transmitter
                    # Creating the seed as a function of position and sub. In
                    # this way users in the same position will experience same
                    # fading coefficient
                    s = np.abs(np.sum((f + c_i.coord + n_i.coord +
                                       c_j.coord + n_j.coord) * 1e4,
                                      dtype=int))
                    if j != i:  # channel is modeled as a Rayleigh fading
                        # Computing Path Loss
                        d = np.linalg.norm(c_i.coord + n_i.coord - c_j.coord - n_j.coord)
                        pl = 20 * np.log10(4 * np.pi * d0 * freq_sc[f] / light_speed) \
                            - n_i.G - n_j.G + 10 * c_i.pl_exp * np.log10(d) \
                            + 10 * ((2 - c_j.pl_exp) * np.log10(d0)
                                    + (c_j.pl_exp - c_i.pl_exp) * np.log10(d0))
                        # Computing Shadowing
                        sh = common.fading("Shadowing", shape=c_i.sh_std, seed=s)
                        # Computing fading matrix
                        fad = common.fading("Rayleigh", seed=s,  dim=(n_i.ant, n_j.ant))
                        # Let the pieces fit
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = \
                            fad * np.sqrt(10 ** (-(pl + sh) / 10))
                    elif n_j.dir == 'FD':  # j == i
                        # Full Duplex fading is Rice distributed
                        fad = common.fading("Rice", dim=(n_i.ant, n_j.ant),
                                            shape=n_j.r_shape, seed=s)
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = \
                            fad * np.sqrt(10 ** (n_i.si_coef / 10))
                    else:  # j == 1 dir != 'FD'
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = \
                            np.zeros((n_i.ant, n_j.ant))
        self.chan_gain = data

    # Visualization methods
    def plot_scenario(self):
        """This method will plot the scenario of communication"""
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif',
                      'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # Open axes
        fig, ax = plt.subplots()
        for c in self.cell:
            # Cell positioning
            cell_out = Circle(c.coord, c.r_outer, facecolor='#45EF0605',
                              edgecolor=(0, 0, 0), ls="--", lw=1)
            cell_in = Circle(c.coord, c.r_inner, facecolor='#37971310',
                             edgecolor=(0, 0, 0), ls="--", lw=0.8)
            ax.add_patch(cell_out)
            ax.add_patch(cell_in)
            # User positioning
            delta = c.r_outer / 100
            plt.text(c.coord[0] + c.r_outer / 2, c.coord[1] + c.r_outer / 2,
                     s=f'cell {c.id}', fontsize=11, c='#D7D7D7')
            for n in c.node:
                plt.scatter(c.coord[0] + n.coord[0], c.coord[1] + n.coord[1],
                            c=dic.color[n.type], marker=dic.mark[n.dir],
                            label=f'{n.type} ({n.dir})')
                plt.text(c.coord[0] + n.coord[0],
                         c.coord[1] + n.coord[1] + delta,
                         s=f'{n.id[1]}', fontsize=11)
            # Plot channel gain link from node to node.useful
            for n in c.node:
                if n.type in dic.user_types:
                    ax = plt.gca()
                    x = c.coord[0] + [n.coord[0], n.useful.coord[0]]
                    y = c.coord[1] + [n.coord[1], n.useful.coord[1]]
                    line = mln.Line2D(x, y, color='#CACACA', linewidth=0.4,
                                      linestyle='--')
                    ax.add_line(line)
        # Set axis
        ax.axis('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Finally
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.show(block=False)

    def show_chan(self, subs: (int, list) = None):
        """Shows the chan_gain in a pretty and readable way.

        Parameters
        __________
        subs : (int, list)
            the subcarriers to be visualized; it can be a single or a list of subcarriers
        """
        # Control if channel gain tensor is built
        if self.chan_gain is None:
            warnings.warn('Channel gain tensor not instantiated.')
            return
        # Control on input
        if subs is None:
            subs = list(range(self.subs))
        elif isinstance(subs, int):
            subs = [subs]
        with np.errstate(divide='ignore'):
            user_grid = 20 * np.log10(np.abs(np.mean(self.chan_gain, axis=(3, 4))))
        out = str()
        nodes = list(range(self.chan_gain.shape[1]))
        for ind, f in enumerate(subs):
            head = [f'f={f}'] + nodes
            out += tabulate(user_grid[f], headers=head, floatfmt=".2f",
                            showindex="always", numalign="right",
                            tablefmt="github")
            out += '\n\n'
        print(out)
