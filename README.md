# GT-DQN
Synergistic combination of hieararchical game theory and deep reinforcement learning.

## Summary
Combination of DQN and level-k reasoning is proposed in this work to model human driver behaviors through driving policies proposals.

## Dependencies

These must be installed before next steps.

+ Python 3.5+
+ Tensorflow >= 2.0


## Training
### 3.1. Training Level 1
Run the python file TrainLevel1.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel1.py
```

At the end of the training, i.e. after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat'. If the reward is converged and no collusion is occured in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level1/'. Else, continue training by uncommenting 136th and 137th lines. 

### 3.2. Training Level 2
Run the python file TrainLevel2.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel2.py
```

At the end of the training, i.e. after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat'. If the reward is converged and no collusion is occured in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level2/'. Else, continue training by uncommenting 136th and 137th lines. 

### 3.3. Training Level 3
Run the python file TrainLevel3.py from the terminal/command window with a script given below or from a python IDE. 

```
python TrainLevel3.py
```

At the end of the training, i.e. after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat'. If the reward is converged and no collusion is occured in recent episodes, move the final weight, i.e. 'datanew/weight49.h5', to the folder 'datanew/level3/'. Else, continue training by uncommenting 136th and 137th lines. 

## Results
### 4.1. Generating Results
In order to understand the performance of the generated policies in terms of modeling human driver behaviors, these policies are compared with real data. For the comparisons, Kolmogorov Goodness-of-Fit Test for Discontinuous Distributions is used.

### 4.2. US101 Dataset 
After obtaining the driving policies, i.e., level-1, level-2, ..., on the ksComparisonRandom100.py file, change lines 88-90 properly by setting the locations of the trained policy weights as the paths. Then, in order to compare the US101 Real Data with the proposed policies, run the python file ksComparisonRandom100.py on the terminal/command windows with the script given below. 

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





