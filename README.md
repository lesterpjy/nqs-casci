## Artificial Neural Networks Applied as Molecular Wave Function Solvers
==============================================================
Neural-network quantum state for exact diagonalization in CAS-CI calculations
<img src="https://github.com/lesterpjy/nqs_casci/blob/master/img/nqs_.png" width="550">

This work published on [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.9b01132) utilize Boltzmann machine (BM) architectures as an encoder of *ab initio* molecular many-electron wave functions represented with the complete active space configuration interaction (CAS-CI) model. This ansatz termed the neural-network quantum state or NQS, was first introduced by Carleo and Troyer in their seminal [paper](https://arxiv.org/abs/1606.02318), and is utilized here for finding a variationally optimal form of the ground-state wave function on the basis of the energy minimization. In addition to RBMs that was implemented in the original paper, we further introduced fully connected BMs, and higher-order BMs to explore convergence to global minima afforded by their concave log-liklihood functions.

The algorithm was implemented with an in-house program suite [ORZ](https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24808) that performs the quantum chemical calculations.
A [snippet](https://github.com/lesterpjy/nqs_casci/blob/master/nqs.cpp) of the code I wrote for this work is provided in this repository, for showcasing purposes.
