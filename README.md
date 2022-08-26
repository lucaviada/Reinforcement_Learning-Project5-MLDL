# Sim-to-Real transfer of Reinforcement Learning policies in robotics - Project5_MLDL

This repository contains the implementation of the concepts described in our work: "titolo paper" for the Machine Learning and Deep Learning course project on Reinforcement Learning in the context of robotic systems.
 
# Description
<img align = "right" src="hopper.png" width="300" height="300">

Reinforcement Learning is considered a new paradigm, different from Supervised and Unsupervised Learning, its strength is represented by the possibility offered to an agent to learn how to behave from its actions.


First of all, this project aims to reproduce some algorithms presented in the state-of-the-art, such as: 

*  REINFORCE with Baseline
*  Actor-Critic
*  Trust Region Policy Optimization (TRPO)
*  Proximal Policy Optimization (PPO)

Moreover, it focuses on the transferring of policies from the simulated world to the real one (that in our case is represented by another simulator, different from the source only for the first mass, shifted of 1 kg).
In order to face the reality gap, due mainly to the difficulty to model parameters of the real world in simulation, concepts like Domain Randomization  (DR) and Adaptive Domain Randomization have been explored. The implementation of the first foresees the randomization of the masses, sampling from Uniform Distributions. Finally, to improve some limits of the DR, an implementation of the Simopt is provided.


# Getting Started
## Dependencies

    Describe any prerequisites, libraries, OS version, etc., needed before installing program.
    ex. Windows 10

## Installing

    How/where to download your program
    Any modifications needed to be made to files/folders

## Executing program

    How to run the program
    Step-by-step bullets

```  code blocks for commands ```





# Authors

* Giovanni Cadau
* Federica Lupo
* Luca Viada


