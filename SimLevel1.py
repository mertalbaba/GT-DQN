import Policy, Car, Params, Action, Message, DQNAgent, State, DRQNAgent
import numpy as np
from numpy import array
import math
import scipy.special as sci

lncount = 0
mndist = 0
mndistcnt = 0
mnspd = 0

if True:
    num_state_resets = 400
    maxcars = 125
    num_episodes = 100
    runtime = 60
    state_size = 19
    action_size = 7
    target_up = 20
    agent = DQNAgent.DQNAgent(state_size, action_size)
    agent.load("datanew/level1/weight43.h5") #44 is 0.4, #46 is 0.2
    agent.T = 1
    done = False
    batch_size = 50
    collusion_count = 0

    for i in range (0, num_state_resets):
        numcars = 100
        state = 0
        state = State.State(numcars, 0)
        check = False
        index = 0


        for j in range(0, num_episodes):
            runsteps = int(runtime / Params.Params.timestep)
            count = 0
            del state.cars
            del state
            state = State.State(numcars, 0)
            state.select_car_positions()

            for step in range(0, runsteps):
                currentstate = state.get_Message(state.cars[0])
                currentstate = np.reshape(currentstate, [1, 1, state_size])
                actionget = agent.act(currentstate)
                if actionget == 5 or actionget == 6:
                    lncount += 1
                mndist += currentstate[0,0,10]
                mndistcnt += 1
                mnspd += state.cars[0].velocity_x
                for temp in state.cars:
                    if temp.equals(state.cars[0]):
                        temp.updateMotionNew(Message.Message(state.get_Message(temp)), state.cars[0],
                                             state.get_numericMessage(temp), False, actionget)
                    else:
                        temp.updateMotion(Message.Message(state.get_Message(temp)), state.cars[0],
                                          state.get_numericMessage(temp),
                                          False)
                done = state.collision_checkandreset_new()
                if done == True:
                    print('mert')
                if done:
                    file = open('datanew/colstates.dat', 'a')
                    file.write(str(currentstate) + '\n')
                    file.close()
                    collusion_count += 1
                    break
                del currentstate, actionget

            print('%' + str(100 * collusion_count / (100*i+j + 1)) + ' Eps: ' + str(100*i+j + 1))
            print('Lane Change '+str(lncount))
            print('Mean Dist. ' + str(mndist/mndistcnt))
            print('Mean Speed ' + str(mnspd / mndistcnt))

        state = None