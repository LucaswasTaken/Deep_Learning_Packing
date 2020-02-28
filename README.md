# Deep_Learning_Packing
An initial aproach for particle packing using LSTM networks. The results and methods are described in the paper "Evaluation Of A Particle Packing Method Using Deep Learning", published at the The XL Ibero-Latin-American Congress on Computational Methods in Engineering (CILAMCE 2019).


This project presents a particle packing approach based on the Long Short-Term Memory
(LSTM) recurrent neural network architecture. An essential task for the simulation of discontinuous
media that use the Discrete Element Method (DEM) is the generation of an initial set of particles, which
represent the discontinuous media of interest, with their corresponding positions and radii. The literature
presents several packing strategies for different particle geometries such as disks (two-dimensional
representation) and spheres (three-dimensional representation). In this context, this paper aims to
evaluate the use of deep models based on the LSTM neural network architecture for particle packing.
The proposed strategy is comprised of the following steps: a) collecting training data from models by
employing any particle packing method, such as the Simple Sequential Inhibition (SSI) or the Poisson
Disk Sampling; b) training several variants of LSTM networks to generate the particles by
experimenting different combinations of hyper-parameters values; c) generate examples through the
proposed algorithm, to evaluate the trained networks. The methodology was implemented using the
Keras and the Tensorflow libraries to build, train and evaluate the neural networks. Examples using the
different network configurations are presented in order to evaluate the accuracy and applicability of the
proposed method
