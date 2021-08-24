import Params, Message, DQNAgent, State
import numpy as np
from numpy import array
import math
import scipy.special as sci
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.optimizers import Adam
import os

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

training = True

if training:
    num_state_resets = 50
    maxcars = 125
    num_episodes = 100
    runtime = 100
    avg_reward = 0
    finpoint = 0
    pol = 0
    state_size = 19
    action_size = 7
    target_up = 20
    # agentlev1 = DQNAgent2.DQNAgent2(state_size, action_size)
    # agentlev1.load("data/level1/weight325.h5")
    # agentlev1.T = 1
    agent = DQNAgent.DQNAgent(state_size, action_size)
    #agent.load('datanew/weight50.h5')
    #agent.T = 1
    done = False
    batch_size = 50


    for i in range (0, num_state_resets):
        numcars = 125
        if i >= 12.5:
            numcars = 100
        if i >= 37.5:
            numcars = 125

        if i == (num_state_resets-1):
            finpoint = 1
        print('State:'+str(i)+' Training with car number: '+str(numcars))
        state = 0
        state = State.State(numcars, pol)
        entropy_states = []
        #avg_reward = 0
        entropy = []
        collusion = []
        entropy_count = 0
        collusion_count = 0
        check = False
        index = 0


        for j in range(0, num_episodes):
            print(100*i+j)
            runsteps = int(runtime / Params.Params.timestep)
            count = 0
            collusion_count = 0
            del state.cars
            del state
            state = State.State(numcars, pol)
            state.select_car_positions()
            reward = 0
            temp_entropy = 0
            sum = 0
            counter = 0
            gamma = 0.9
            value_buffer = 0.1
            numvisits = []
            for step in range(0, runsteps):

                currentstate = state.get_Message(state.cars[0])
                currentstate = np.reshape(currentstate, [1, 1, state_size])
                actionget = agent.act(currentstate)
                for temp in state.cars:
                    if temp.equals(state.cars[0]):
                        temp.updateMotionNew(Message.Message(state.get_Message(temp)), state.cars[0], state.get_numericMessage(temp), False, actionget)
                    else:
                        temp.updateMotion(Message.Message(state.get_Message(temp)), state.cars[0], state.get_numericMessage(temp),
                                             False)
                        # curr = state.get_intMessage2(temp)
                        # curr = np.reshape(curr, [1, 1, state_size])
                        # actiongetted = agentlev1.act(curr)
                        # temp.updateMotionNew(state.get_Message(temp), state.cars[0], state.get_numericMessage(temp), False, actiongetted)
                        # del curr
                        # del actiongetted

                current_act = state.cars[0].current_action
                reward = state.get_reward()
                sum += reward
                counter += 1
                nextstate = state.get_Message(state.cars[0])
                nextstate = np.reshape(nextstate, [1, 1,state_size])
                done = state.collision_checkandreset_new()
                #this is made to prevent examples which may be misinforming
                if state.cars[0].velocity_x >= Params.Params.min_speed and state.cars[0].velocity_x <= Params.Params.max_speed:
                    agent.remember(currentstate, actionget, reward, nextstate, done)
                avg_reward += (reward - avg_reward) / float((step + 1 + j * runsteps + i * num_episodes * runsteps))
                #print('avg_reward: ' + str(avg_reward) + ' ', end=' ')
                if len(agent.memory) > 2*batch_size:
                    agent.replay(batch_size)
                if done:
                    print('collision at step ' + str(step) + ' ', end = ' ')
                    collusion_count += 1
                    del currentstate, current_act, actionget
                    break
                del currentstate, current_act, actionget

            print('', end='\n')


            if j == (num_episodes - 1):
                check = True;

            if j%target_up == 0:
                agent.update_target_model()
            avg = sum/counter
            #print(avg, end='\n')
            collusion.append(collusion_count)
            file = open('datanew/reward.dat', 'a')
            file.write('' + str((j + i * num_episodes)) + '\t' + str(avg_reward) + '\n')
            file.close()
            file = open('datanew/reward2.dat', 'a')
            file.write('' + str((j + i * num_episodes)) + '\t' + str(avg) + '\n')
            file.close()
            file = open('datanew/collusion.dat', 'a')
            file.write(str(collusion_count) + '\n')
            file.close()
            # file = open('data/visits.dat', 'a')
            # for item in numvisits:
            #     file.write(str(item) + '\n')
            # file.close()

        state = None
        if agent.T > 1:
            agent.T = agent.T*0.925
            #agent.T = agent.T*0.984474
        fname = './datanew/weight'+str(i)+'.h5'
        agent.save(fname)















