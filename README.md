# UoA_MastersWorking
A repository for investigating the Julia implementation of profile likelihood and uncertainty quantification/propogation for mechanistic models

Contains files used to fulfil the requirements of a Masters of Engineering at The University of Auckland by Joel Trent between March 2023 and February 2024.

The files used to construct the outputs within the thesis are contained in the _Bespoke graphics_ and _Experiments_ folders. _Experiments/Models_ contains the model implementations as required by [LikelihoodBasedProfileWiseAnalysis](https://github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl), while _Experiments/Outputs_ contains the outputs of each experiment. Figures are created using _Experiments/Experiments.ipynb_. Julia files within _Experiments_ contain the code used to run the experiments; the C?_ prepended to each file gives the first section that model is used in. The STAT5 and stochastic models were not used within the main body of the thesis but are kept as examples for use in future work.

Supervised by Oliver Maclaren, Ruanui Nicholson and Matthew Simpson.
