#!/usr/bin/env python3
# filename "common_dict.py"

# Global dictionaries
# Two main node types are defined: base stations and users.
# For each 2 subtypes are possible.
# The definition is the following.
bs_types = {'mBS': 0, 'fBS': 1}
user_types = {'UE': 2, 'D2D': 3}
node_types = dict(bs_types, **user_types)
# The possible directions for the nodes are the following.
bs_dir = {'HD': 0, 'FD': 1}
user_dir = {'UL': 2, 'DL': 3}
node_dir = dict(bs_dir, **user_dir)
# The supported traffic are the following
traffic_types = {'eMBB': 0, 'URLLC': 1}  # , 'mMTC': 2}
# The supported channel types are the following.
channel_types = {'AWGN', 'Rayleigh', 'Rice', 'Shadowing'}

# The following are defined for graphic purpose only
color = {'mBS': '#DC2516', 'fBS': '#F57707',
         'UE': '#36F507', 'D2D': '#0F4EEA'}
mark = {'HD': '^', 'FD': 'v', 'UL': 'o', 'DL': 'x'}
