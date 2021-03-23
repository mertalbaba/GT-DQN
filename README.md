# Continuous DQN
Autonomous Driving Policy through Deep Q-Networks

## Summary
Combination of DQN and level-k reasoning is proposed in this work to model human driver behaviors through driving policies.oposals.

## Dependencies

These must be installed before next steps.

+ Python 3.5+
+ Tensorflow >= 2.0


## Training
### 3.1. Training Level 1
Run the python file LaneChangeLev1.py from the terminal/command window as
```
python LaneChangeLev1.py
```
or from the IDE. 

At the end of the training, i.e. after 5000 episodes, analyze the reward and collusion statistics by inspecting 'datanew/reward.dat' and 'datanew/collusion.dat'. If the reward is converged and no collusion is occured in recent episodes, move the final weight, i.e. 'datanew/weight50.h5', to the folder 'datanew/level1/'. Else, continue training by uncommenting 136th and 137th lines. 

### 3.2. Training Level 2


### 3.3. Training Level 3

## Results
### 4.1. Generating Results

### 4.2. US101 Dataset                                                                                                                                                                                                                                                                                                         |

### 4.3. I80 Dataset

## Additional
### Brief Description of each file
