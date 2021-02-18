Deep Reinforcement Learning for URLLC data management on top of scheduled eMBB traffic
**************************************************************************************

Fabio Saggese, Luca Pasqualini, Marco Moretti, Andrea Abrardo
#############################################################

GitHub for a resource allocation research project through reinforcement learning.

A proper physical layer resources allocation strategy of different kinds of traffic is key to efficient and reliable beyond 5G networks.

We engineered a simulated environment, where the task is to manage the slicing problem between ultra-reliable low-latency (URLLC) and enhanced Mobile BroadBand (eMBB) traffics.
For the simulation, we consider a time-frequency resource grid populated with eMBB codewords.

The algorithm used is Proximal Policy Optimization (PPO) with rewards-to-go and Generalized Advantage Estimation (GAE-Lambda) buffer.
Within the simulation, we use PPO to train an agent to dynamically allocate the incoming URLLC traffic on top of the eMBB occupied resources by means of puncturing.

Additional information about the current approach can be found in the `arXiv article <TODO>`_.

**License**

The same of the article.

**Framework used**

- To execute reinforcement learning the framework `USienaRL <https://github.com/InsaneMonster/USienaRL>`_ (`usienarl package <https://pypi.org/project/usienarl/>`_) is used.

**Compatible with Usienarl v0.7.9**

**Backend**

- *Python 3.6*
- *Tensorflow 1.15*