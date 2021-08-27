# GT-DQN
The synergistic combination of hierarchical game theory and deep reinforcement learning for modeling human driver behavior

## Summary
A combination of DQN and level-k reasoning is proposed in this work to model human driver behaviors through driving policies proposals.

## Dependencies

These must be installed before the next steps.

+ Python 3.5+
+ Tensorflow >= 2.0


## Training
### 3.1. Training Level 1
Run the python file TrainLevel1.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel1.py
```

At the end of the training, i.e., after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat.' If the reward is converged and no collusion has occurred in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level1/'. Else, continue training by uncommenting 136th and 137th lines. 

### 3.2. Training Level 2
Run the python file TrainLevel2.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel2.py
```

At the end of the training, i.e., after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat.' If the reward is converged and no collision has occurred in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level2/'. Else, continue training by uncommenting 136th and 137th lines. 

### 3.3. Training Level 3
Run the python file TrainLevel3.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel3.py
```

At the end of the training, i.e., after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat.' If the reward is converged and no collusion has occurred in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level3/'. Else, continue training by uncommenting 136th and 137th lines. 

## Results
### 4.1. Generating Results
In order to understand the performance of the generated policies in modeling human driver behaviors, these policies are compared with real data. For the comparisons, the Kolmogorov Goodness-of-Fit Test for Discontinuous Distributions is used.

### 4.2. US101 Dataset 
After obtaining the driving policies, i.e., level-1, level-2, ..., on the ksComparisonRandom100.py file, change lines 88-90 properly by setting the locations of the trained policy weights as the paths. Then, to compare the US101 Real Data with the proposed policies, run the python file ksComparisonRandom100.py on the terminal/command windows with the script given below. 

```
python ksComparisonRandom100.py
```

### 4.3. I80 Dataset
After obtaining the driving policies, i.e., level-1, level-2, ..., on the ksComparisonRandom100_i80.py file, change lines 88-90 properly by setting the locations of the trained policy weights as the paths. Then, in order to compare the US101 Real Data with the proposed policies, run the python file ksComparisonRandom100_i80.py on the terminal/command windows with the script given below. 

```
python ksComparisonRandom100_i80.py
```

## Appendix
### Action.py
All possible actions, i.e., actions in the action space, are defined in this file.

### Car.py
Vehicle class is implemented in this file. Position and velocity update functions are defined. 

### DQNAgent.py
A model of the utilized RL approach, DQN, is defined in this file. Experience replay, target network, action prediction are implemented through functions.

### Message.py
Helper class for the observation space. 

### Params.py
Utilized parameter values are defined in this file.

### SimLevel1.py, SimLevel2.py, SimLevel3.py
Simulators are implemented in these files, in which a level-k agent is placed in an environment where all the drivers are level-(k-1).

### TrainLevel1.py, TrainLevel2.py, TrainLevel3.py
Training code for obtaining a level-k policy is implemented in these files. 

### ksComparisonRandom100.py
The Kolmogorov Smirnov test for discontinuous distributions is implemented in this file and utilized to compare the US101 real data and the proposed policies. 

### ksComparisonRandom100_i80.py
The Kolmogorov Smirnov test for discontinuous distributions is implemented in this file and utilized to compare the I80 real data and the proposed policies. 
