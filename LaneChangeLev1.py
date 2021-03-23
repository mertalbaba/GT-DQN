import Policy, Car, Params, Action, Message, DQNAgent, State, DQNAgent2
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
zerogen = False
comparison = False
newcomp = False
kscomp = False
simulation = False

def get_action_index(act):
    if act.action == 'maintain':
        return 0
    elif act.action == 'accelerate':
        return 1
    elif act.action == 'decelerate':
        return 2
    elif act.action == 'hard_accelerate':
        return 3
    elif act.action == 'hard_decelerate':
        return 4
    elif act.action == 'move left':
        return 5
    elif act.action == 'move right':
        return 6
    else:
        print('Error in get_action_index for action: '+act.action)
        return -1


def int_to_msg(num, lane):
    base3 = int2base(num, 3)
    base3 = '{:18s}'.format(base3)
    j = base3.replace(' ', '0')
    base3 = j
    msgwords = []
    for i in range(0,18,2):
        msgwords.append(d_word(base3[i]))
        msgwords.append(v_word(base3[i]))
    msgwords.append(lane_word(str(lane)))
    return Message.Message(msgwords)

def int2base(x, b, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    'convert an integer to its string representation in a given base'
    if b < 2 or b > len(alphabet):
        if b == 64:  # assume base64 rather than raise error
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        else:
            raise AssertionError("int2base base out of range")
    if isinstance(x, complex):  # return a tuple
        return (int2base(x.real, b, alphabet), int2base(x.imag, b, alphabet))
    if x <= 0:
        if x == 0:
            return alphabet[0]
        else:
            return '-' + int2base(-x, b, alphabet)
    # else x is non-negative real
    rets = ''
    while x > 0:
        x, idx = divmod(x, b)
        rets = alphabet[idx] + rets
    return rets

def d_word(i):
    if(i == '0'):
        return 'close'
    elif (i =='1'):
        return 'nominal'
    elif (i == '2'):
        return 'far'
    else:
        print('cant find '+i+' in d_word() options')
        return '*'

def v_word(i):
    if(i == '0'):
        return 'stable'
    elif (i =='1'):
        return 'approaching'
    elif (i == '2'):
        return 'retreating'
    else:
        print('cant find '+i+' in v_word() options')
        return '*'

def lane_word(i):
    if(i == '0'):
        return 'right'
    elif (i =='1'):
        return 'center1'
    elif (i == '2'):
        return 'center2'
    elif (i == '3'):
        return 'center3'
    elif (i == '4'):
        return 'left'
    else:
        print('cant find '+i+' in lane_word() options')
        return '*'


def read_integers(filename):
    with open(filename) as f:
        return map(int, f)

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
    #agent.load('data/level1new/weight74.h5')
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
            agent.T = agent.T*0.968
            #agent.T = agent.T*0.984474
        fname = './datanew/weight'+str(i)+'.h5'
        agent.save(fname)

elif kscomp:
    print('starting comp')
    state_size = 19
    action_size = 7
    agentl1 = DQNAgent.DQNAgent(state_size, action_size)
    agentl2 = DQNAgent.DQNAgent(state_size, action_size)
    agentl3 = DQNAgent.DQNAgent(state_size, action_size)
    agentl1.load("dqn_weights/level1/weight.h5")
    agentl2.load("dqn_weights/level2/weight.h5")
    agentl3.load("dqn_weights/level3/weight.h5")
    agentl1.T = 1
    agentl2.T = 1
    agentl3.T = 1
    with open('data/realdata.txt') as f:
        realdata = [[float(x) for x in line.split()] for line in f]

    index = 0;
    currentcar = 2;
    sss1 = 0;
    sss2 = 0;
    sss3 = 0;
    sssd = 0;
    cnt = 0;
    cnt1 = 0;
    cnt2 = 0;
    cnt3 = 0;
    indexcomp = 1;

    visited = 0;
    visited1 = 0;
    visited2 = 0;
    visited3 = 0;
    allvisited = 0;

    state = State.State(2, 0)

    index3 = -1
    index2 = -1
    resultcomp3 = []
    resultcomp2 = []

    i = 0

    while i < len(realdata):
        if (realdata[i][0] == currentcar):
            if (realdata[i][10] > 2):
                #print(i)
                allvisited = allvisited + 1;
                crtval = 12.59;
                rd = 0;
                pol = 0;
                pol1 = 0;
                pol2 = 0;
                pol3 = 0;
                dumb = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]);
                vis1 = 100#visits1[int(realdata[i][1])]
                vis2 = 100#visits2[int(realdata[i][1])]
                vis3 = 100#visits3[int(realdata[i][1])]
                visreal = realdata[i][10]

                lane = int(realdata[i][1] % 5)
                num = int(realdata[i][1] / 5)
                numl = lane
                msg = int_to_msg(num, lane)
                currentstate = None
                currentstate = state.get_intMessage3(msg)
                currentstate = np.reshape(currentstate, [1, 1, state_size])
                pol1 = pol2 = pol3 = None
                pol1 = agentl1.getq_prob(currentstate)
                #pol1[0] = np.nan_to_num(pol1[0])
                pol2 = agentl2.getq_prob(currentstate)
                #pol2[0] = np.nan_to_num(pol2[0])
                pol3 = agentl3.getq_prob(currentstate)
                #pol3[0] = np.nan_to_num(pol3[0])
                rd = np.zeros((1, 7))

                p1ind = []
                p2ind = []
                p3ind = []
                rdind = []
                p1sum = 0
                p2sum = 0
                p3sum = 0
                rdsum = 0
                for p1 in range(7):
                    if (pol1[0, p1] < 0.01):
                        pol1[0, p1] = 0.01
                        p1sum += 0.01
                    else:
                        p1ind.append(p1)
                    if (pol2[0, p1] < 0.01):
                        pol2[0, p1] = 0.01
                        p2sum += 0.01
                    else:
                        p2ind.append(p1)
                    if (pol3[0, p1] < 0.01):
                        pol3[0, p1] = 0.01
                        p3sum += 0.01
                    else:
                        p3ind.append(p1)

                    rd[0, p1] = realdata[i][p1 + 2]/realdata[i][10]
                    if (rd[0, p1] < 0.01):
                        rd[0, p1] = 0.01
                        rdsum += 0.01
                    else:
                        rdind.append(p1)

                p1sum = 1-p1sum
                p2sum = 1-p2sum
                p3sum = 1-p3sum
                rdsum = 1-rdsum

                cntind = 0
                for cntind in range(len(p1ind)):
                    pol1[0,p1ind[cntind]] *= p1sum

                cntind = 0
                for cntind in range(len(p2ind)):
                    pol2[0, p2ind[cntind]] *= p2sum

                cntind = 0
                for cntind in range(len(p3ind)):
                    pol3[0, p3ind[cntind]] *= p3sum

                cntind = 0
                for cntind in range(len(rdind)):
                    rd[0, rdind[cntind]] *= rdsum

                # pol1 = pol1/(np.sum(pol1))
                # pol2 = pol2/(np.sum(pol2))
                # pol3 = pol3/(np.sum(pol3))
                # rd = rd/(np.sum(rd))

                if ((vis1 > 0) or (vis2 > 0) or (vis3 > 0)):
                    dtemp = 0;
                    resultcomp3.append([])
                    index3 += 1
                    resultcomp3[indexcomp - 1].append(currentcar)
                    resultcomp3[indexcomp - 1].append(realdata[i][1])
                    resultcomp3[indexcomp - 1].append(50)
                    resultcomp3[indexcomp - 1].append(50)
                    resultcomp3[indexcomp - 1].append(50)
                    resultcomp3[indexcomp - 1].append(50)

                    sn = np.zeros((1,7));
                    h = np.zeros((1,7)); # H(x)
                    hp = np.array([[1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]]);
                    snp = rd;

                    sn[0, 0] = snp[0, 0];  # Sn(x)
                    h[0, 0] = hp[0, 0];

                    ind1 = 1;
                    ind2 = 1;

                    for p1 in range(1, 7):
                        sn[0, p1] = sn[0, p1 - 1] + snp[0, p1];
                        if sn[0, p1] > 1:
                            sn[0, p1] = 1
                        h[0, p1] = h[0, p1 - 1] + (hp[0, p1]);
                        if h[0, p1] > 1:
                            h[0, p1] = 1

                    nsample = visreal;

                    temp = np.abs(sn - h);
                    temp2 = sn - h;
                    temp3 = h - sn;

                    d = np.max(temp);
                    dminus = np.max(temp3);
                    dplus = np.max(temp2);
                    dminus = d;
                    dplus = d;

                    limitminus = nsample * (1 - dminus);
                    limitplus = nsample * (1 - dplus);

                    carrminus = np.zeros(int(limitminus))
                    barrminus = np.zeros(int(limitminus))
                    carrplus = np.zeros(int(limitplus))
                    barrplus = np.zeros(int(limitplus))

                    # calculating c values for d -
                    for j in range(int(limitminus)):
                        tempvalminus = dminus + (j / nsample);
                        okcmin = 0;
                        for tempi in range(7):
                            if (h[0, tempi] >= tempvalminus):
                                if (okcmin == 0):
                                    okcmin = 1
                                    carrminus[j] = (1 - h[0, tempi])

                        if (okcmin == 0):
                            carrminus[j] = (1 - tempvalminus);

                    # calculating b values for d -
                    for j in range(int(limitminus)):
                        if (carrminus[j] > 0):
                            barrminus[j] = 1
                            for tempcount in range(j):
                                barrminus[j] = barrminus[j] - (
                                            sci.comb(j, tempcount) * (pow(carrminus[tempcount], (j - tempcount))) *
                                            barrminus[tempcount]);

                    # calculating critical value of d -
                    cvminus = 0;
                    for j in range(int(limitminus)):
                        cvminus = cvminus + (sci.comb(nsample, j) * (pow(carrminus[j], (nsample - j))) * barrminus[j]);

                    # calculating c values(f) for d +
                    for j in range(int(limitplus)):
                        tempvalplus = 1 - dplus - (j / nsample)
                        okcplus = 0;
                        for tempi in range(7):
                            if (h[0, tempi] <= tempvalplus):
                                if (okcplus == 0):
                                    carrplus[j] = (h[0, tempi])
                                    okcplus = 1

                        if (okcplus == 0):
                            carrplus[j] = (tempvalplus);

                    # calculating b values(e) for d +
                    for j in range(int(limitplus)):
                        if (carrplus[j] > 0):
                            barrplus[j] = 1
                            for tempcount in range(j):
                                barrplus[j] -= (sci.comb(j, tempcount) * (pow(carrplus[tempcount], (j - tempcount))) *
                                                barrplus[tempcount])

                    # calculating critical value of d +
                    cvplus = 0;
                    for j in range(int(limitplus)):
                        cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j], (nsample - j))) * barrplus[j])

                    cv = cvplus + cvminus;
                    # result = pdsum + pdminus;
                    if (cv > 0.05):
                        sssd = sssd + 1

                    resultcomp3[indexcomp-1][5] = cv

                if ((vis1 > 0)):
                    dtemp = 0;
                    sn = np.zeros((1,7));
                    h = np.zeros((1,7)); # H(x)
                    hp = pol1
                    snp = rd;

                    sn[0, 0] = snp[0, 0];  # Sn(x)
                    h[0, 0] = hp[0, 0];

                    ind1 = 1;
                    ind2 = 1;

                    for p1 in range(1, 7):
                        sn[0, p1] = sn[0, p1 - 1] + snp[0, p1];
                        if sn[0, p1] > 1:
                            sn[0, p1] = 1
                        h[0, p1] = h[0, p1 - 1] + (hp[0, p1]);
                        if h[0, p1] > 1:
                            h[0, p1] = 1

                    nsample = visreal;

                    temp = np.abs(sn - h);
                    temp2 = sn - h;
                    temp3 = h - sn;

                    d = np.max(temp);
                    dminus = np.max(temp3);
                    dplus = np.max(temp2);
                    dminus = d;
                    dplus = d;

                    limitminus = nsample * (1 - dminus);
                    limitplus = nsample * (1 - dplus);

                    carrminus = np.zeros(int(limitminus))
                    barrminus = np.zeros(int(limitminus))
                    carrplus = np.zeros(int(limitplus))
                    barrplus = np.zeros(int(limitplus))

                    # calculating c values for d -
                    for j in range(int(limitminus)):
                        tempvalminus = dminus + (j / nsample);
                        okcmin = 0;
                        for tempi in range(7):
                            if (h[0, tempi] >= tempvalminus):
                                if (okcmin == 0):
                                    okcmin = 1
                                    carrminus[j] = (1 - h[0, tempi])

                        if (okcmin == 0):
                            carrminus[j] = (1 - tempvalminus);

                    # calculating b values for d -
                    for j in range(int(limitminus)):
                        if (carrminus[j] > 0):
                            barrminus[j] = 1
                            for tempcount in range(j):
                                barrminus[j] = barrminus[j] - (
                                            sci.comb(j, tempcount) * (pow(carrminus[tempcount], (j - tempcount))) *
                                            barrminus[tempcount]);

                    # calculating critical value of d -
                    cvminus = 0;
                    for j in range(int(limitminus)):
                        cvminus = cvminus + (sci.comb(nsample, j) * (pow(carrminus[j], (nsample - j))) * barrminus[j]);

                    # calculating c values(f) for d +
                    for j in range(int(limitplus)):
                        tempvalplus = 1 - dplus - (j / nsample)
                        okcplus = 0;
                        for tempi in range(7):
                            if (h[0, tempi] <= tempvalplus):
                                if (okcplus == 0):
                                    carrplus[j] = (h[0, tempi])
                                    okcplus = 1

                        if (okcplus == 0):
                            carrplus[j] = (tempvalplus);

                    # calculating b values(e) for d +
                    for j in range(int(limitplus)):
                        if (carrplus[j] > 0):
                            barrplus[j] = 1
                            for tempcount in range(j):
                                barrplus[j] -= (sci.comb(j, tempcount) * (pow(carrplus[tempcount], (j - tempcount))) *
                                                barrplus[tempcount])

                    # calculating critical value of d +
                    cvplus = 0;
                    for j in range(int(limitplus)):
                        cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j], (nsample - j))) * barrplus[j])

                    cv = cvplus + cvminus;
                    # result = pdsum + pdminus;
                    if (cv > 0.05):
                        sss1 = sss1 + 1

                    resultcomp3[indexcomp - 1][2] = cv
                    cnt1 = cnt1 + 1;
                    visited1 = visited1 + 1;


                if ((vis2 > 0)):
                    hdtemp = 0;
                    sn = np.zeros((1, 7));
                    h = np.zeros((1, 7));  # H(x)
                    hp = pol2
                    snp = rd

                    sn[0, 0] = snp[0, 0];  # Sn(x)
                    h[0, 0] = hp[0, 0];

                    ind1 = 1;
                    ind2 = 1;

                    for p1 in range(1, 7):
                        sn[0, p1] = sn[0, p1 - 1] + snp[0, p1];
                        if sn[0, p1] > 1:
                            sn[0, p1] = 1
                        h[0, p1] = h[0, p1 - 1] + (hp[0, p1]);
                        if h[0, p1] > 1:
                            h[0, p1] = 1

                    nsample = visreal;

                    temp = np.abs(sn - h);
                    temp2 = sn - h;
                    temp3 = h - sn;

                    d = np.max(temp);
                    dminus = np.max(temp3);
                    dplus = np.max(temp2);
                    dminus = d;
                    dplus = d;

                    limitminus = nsample * (1 - dminus);
                    limitplus = nsample * (1 - dplus);

                    carrminus = np.zeros(int(limitminus))
                    barrminus = np.zeros(int(limitminus))
                    carrplus = np.zeros(int(limitplus))
                    barrplus = np.zeros(int(limitplus))

                    # calculating c values for d -
                    for j in range(int(limitminus)):
                        tempvalminus = dminus + (j / nsample);
                        okcmin = 0;
                        for tempi in range(7):
                            if (h[0, tempi] >= tempvalminus):
                                if (okcmin == 0):
                                    okcmin = 1
                                    carrminus[j] = (1 - h[0, tempi])

                        if (okcmin == 0):
                            carrminus[j] = (1 - tempvalminus);

                    # calculating b values for d -
                    for j in range(int(limitminus)):
                        if (carrminus[j] > 0):
                            barrminus[j] = 1
                            for tempcount in range(j):
                                barrminus[j] = barrminus[j] - (
                                            sci.comb(j, tempcount) * (pow(carrminus[tempcount], (j - tempcount))) *
                                            barrminus[tempcount]);

                    # calculating critical value of d -
                    cvminus = 0;
                    for j in range(int(limitminus)):
                        cvminus = cvminus + (sci.comb(nsample, j) * (pow(carrminus[j], (nsample - j))) * barrminus[j]);

                    # calculating c values(f) for d +
                    for j in range(int(limitplus)):
                        tempvalplus = 1 - dplus - (j / nsample)
                        okcplus = 0;
                        for tempi in range(7):
                            if (h[0, tempi] <= tempvalplus):
                                if (okcplus == 0):
                                    carrplus[j] = (h[0, tempi])
                                    okcplus = 1

                        if (okcplus == 0):
                            carrplus[j] = (tempvalplus);

                    # calculating b values(e) for d +
                    for j in range(int(limitplus)):
                        if (carrplus[j] > 0):
                            barrplus[j] = 1
                            for tempcount in range(j):
                                barrplus[j] -= (sci.comb(j, tempcount) * (pow(carrplus[tempcount], (j - tempcount))) *
                                                barrplus[tempcount])

                    # calculating critical value of d +
                    cvplus = 0;
                    for j in range(int(limitplus)):
                        cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j], (nsample - j))) * barrplus[j])

                    cv = cvplus + cvminus;
                    # result = pdsum + pdminus;
                    if (cv > 0.05):
                        sss2 = sss2 + 1

                    resultcomp3[indexcomp - 1][3] = cv
                    cnt2 = cnt2 + 1;
                    visited2 = visited2 + 1;


                if ((vis3 > 0)):
                    hdtemp = 0;
                    sn = np.zeros((1, 7));
                    h = np.zeros((1, 7));  # H(x)
                    hp = pol3
                    snp = rd

                    sn[0,0] = snp[0,0];  # Sn(x)
                    h[0,0] = hp[0,0];

                    ind1 = 1;
                    ind2 = 1;

                    for p1 in range (1,7):
                        sn[0,p1] = sn[0,p1-1] + snp[0, p1];
                        if sn[0,p1] > 1:
                            sn[0, p1] = 1
                        h[0,p1] = h[0,p1-1] + (hp[0,p1]);
                        if h[0,p1] > 1:
                            h[0, p1] = 1

                    nsample = visreal;

                    temp = np.abs(sn - h);
                    temp2 = sn - h;
                    temp3 = h - sn;

                    d = np.max(temp);
                    dminus = np.max(temp3);
                    dplus = np.max(temp2);
                    dminus = d;
                    dplus = d;

                    limitminus = nsample * (1 - dminus);
                    limitplus = nsample * (1 - dplus);

                    carrminus = np.zeros(int(limitminus))
                    barrminus = np.zeros(int(limitminus))
                    carrplus = np.zeros(int(limitplus))
                    barrplus = np.zeros(int(limitplus))

                    # calculating c values for d -
                    for j in range(int(limitminus)):
                        tempvalminus = dminus + (j / nsample);
                        okcmin = 0;
                        for tempi in range(7):
                            if (h[0,tempi] >= tempvalminus):
                                if(okcmin == 0):
                                    okcmin = 1
                                    carrminus[j]=(1 - h[0,tempi])

                        if (okcmin == 0):
                            carrminus[j]=(1 - tempvalminus);

                    # calculating b values for d -
                    for j in range(int(limitminus)):
                        if (carrminus[j] > 0):
                            barrminus[j] = 1
                            for tempcount in range(j):
                                barrminus[j] = barrminus[j] - (sci.comb(j, tempcount) * (pow(carrminus[tempcount], (j - tempcount))) * barrminus[tempcount]);

                    # calculating critical value of d -
                    cvminus = 0;
                    for j in range(int(limitminus)):
                        cvminus = cvminus + (sci.comb(nsample, j) * (pow(carrminus[j], (nsample - j))) * barrminus[j]);

                    # calculating c values(f) for d +
                    for j in range(int(limitplus)):
                        tempvalplus = 1 - dplus - (j/nsample)
                        okcplus = 0;
                        for tempi in range(7):
                            if (h[0,tempi] <= tempvalplus):
                                if(okcplus == 0):
                                    carrplus[j] = (h[0,tempi])
                                    okcplus = 1

                        if (okcplus == 0):
                            carrplus[j]=(tempvalplus);

                    # calculating b values(e) for d +
                    for j in range(int(limitplus)):
                        if (carrplus[j] > 0):
                            barrplus[j] = 1
                            for tempcount in range(j):
                                barrplus[j] -= (sci.comb(j, tempcount) * (pow(carrplus[tempcount], (j - tempcount)))*barrplus[tempcount])

                    # calculating critical value of d +
                    cvplus = 0;
                    for j in range(int(limitplus)):
                        cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j], (nsample - j))) * barrplus[j])

                    cv = cvplus + cvminus;
                    # result = pdsum + pdminus;
                    if (cv > 0.05):
                        sss3 = sss3 + 1

                    resultcomp3[indexcomp - 1][4] = cv
                    cnt3 = cnt3 + 1;
                    visited3 = visited3 + 1;



                if ((vis1 > 0) or (vis2 > 0) or (vis3 > 0)):
                    visited = visited + 1;
                    cnt = cnt + 1;
                    indexcomp = indexcomp + 1;
                    check = 1;

        else:
            print(i)
            if cnt1 > 0:
                sss1 = sss1 / cnt1
            else:
                sss1 = -1
            if cnt2 > 0:
                sss2 = sss2 / cnt2
            else:
                sss2 = -1
            if cnt3 > 0:
                sss3 = sss3 / cnt3
            else:
                sss3 = -1
            if cnt > 0:
                sssd = sssd / cnt
            else:
                sssd = -1

            resultcomp2.append([])
            resultcomp2[index].append(currentcar)
            resultcomp2[index].append(sss1)
            resultcomp2[index].append(sss2)
            resultcomp2[index].append(sss3)
            resultcomp2[index].append(sssd)
            index = index + 1;

            chisq1 = 0;
            chisq2 = 0;
            chisq3 = 0;

            currentcar = realdata[i][0]
            i = i - 1

            cnt = 0;
            cnt1 = 0;
            cnt2 = 0;
            cnt3 = 0;

            sss1 = 0;
            sss2 = 0;
            sss3 = 0;
            sssd = 0;
            check = 0;

        i = i + 1;

    rescomp2tmp = np.asarray(resultcomp2)
    rescomp3tmp = np.asarray(resultcomp3)
    np.savetxt('results/old/1/resultcomp2.txt', rescomp2tmp, delimiter='\t')
    np.savetxt('results/old/1/resultcomp3.txt', rescomp3tmp, delimiter='\t')

# elif kscomp:
#     state_size = 19
#     action_size = 7
#     agentl1 = DQNAgent.DQNAgent(state_size, action_size)
#     agentl2 = DQNAgent.DQNAgent(state_size, action_size)
#     agentl3 = DQNAgent.DQNAgent(state_size, action_size)
#     agentl1.load("./level1/weight.h5")
#     agentl2.load("./level2/weight.h5")
#     agentl3.load("./level2/weight.h5")
#     with open('./data/realdata.txt') as f:
#         realdata = [[float(x) for x in line.split()] for line in f]
#
#     with open('level1/visits.dat', 'r+') as f:
#         lines = f.read().splitlines()
#     visits1 = np.zeros(1937102445, dtype=int)
#     for item in lines:
#         ind = int(item)
#         visits1[ind] += 1
#
#     with open('level2/visits.dat', 'r+') as f:
#         lines = f.read().splitlines()
#     visits2 = np.zeros(1937102445, dtype=int)
#     for item in lines:
#         ind = int(item)
#         visits2[ind] += 1
#
#     with open('data/visits.dat', 'r+') as f:
#         lines = f.read().splitlines()
#     visits3 = np.zeros(1937102445, dtype=int)
#     for item in lines:
#         ind = int(item)
#         visits3[ind] += 1
#
#     index = 1;
#     currentcar = 2;
#     sss1 = 0;
#     sss2 = 0;
#     sss3 = 0;
#     sssd = 0;
#     cnt = 0;
#     cnt1 = 0;
#     cnt2 = 0;
#     cnt3 = 0;
#     indexcomp = 1;
#
#     visited = 0;
#     visited1 = 0;
#     visited2 = 0;
#     visited3 = 0;
#     allvisited = 0;
#
#     state = State.State(2, 0)
#
#     index3 = -1
#     index2 = -1
#     resultcomp3 = []
#     resultcomp2 = []
#
#     i = 0
#
#     while i < len(realdata):
#         if (realdata[i][0] == currentcar)
#             if (realdata[i][10] > 0):
#
#                 allvisited = allvisited + 1;
#
#                 crtval = 12.59;
#                 rd = 0;
#                 pol = 0;
#                 pol1 = 0;
#                 pol2 = 0;
#                 pol3 = 0;
#                 dumb = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]);
#                 vis1 = 100  # visits1[int(realdata[i][1])]
#                 vis2 = 100  # visits2[int(realdata[i][1])]
#                 vis3 = 100  # visits3[int(realdata[i][1])]
#                 visreal = realdata[i][10]
#
#                 lane = int(realdata[i][1] % 5)
#                 num = int(realdata[i][1] / 5)
#                 msg = int_to_msg(num, lane)
#                 currentstate = state.get_intMessage3(msg)
#                 currentstate = np.reshape(currentstate, [1, 1, state_size])
#
#                 pol1 = agentl1.getq_prob(currentstate)
#                 pol2 = agentl2.getq_prob(currentstate)
#                 pol3 = agentl3.getq_prob(currentstate)
#                 rd = np.zeros((1, 7))
#
#                 for p1 in range(7):
#                     if (pol1[0, p1] < 0.015):
#                         pol1[0, p1] = 0.015
#                     if (pol2[0, p1] < 0.015):
#                         pol2[0, p1] = 0.015
#                     if (pol3[0, p1] < 0.015):
#                         pol3[0, p1] = 0.015
#
#                     rd[0, p1] = realdata[i][p1 + 2]/realdata[i][10]
#                     if (rd[0, p1] < 0.015):
#                         rd[0, p1] = 0.015
#
#                 pol1 = pol1/(np.sum(pol1))
#                 pol2 = pol2/(np.sum(pol2))
#                 pol3 = pol3/(np.sum(pol3))
#                 rd = rd/(np.sum(rd))
#
#                 if ((vis1 > 0) or (vis2 > 0) or (vis3 > 0)):
#                     dtemp = 0;
#                     resultcomp3.append([])
#                     index3 += 1
#                     resultcomp3[indexcomp - 1].append(currentcar)
#                     resultcomp3[indexcomp - 1].append(realdata[i][1])
#                     resultcomp3[indexcomp - 1].append(50)
#                     resultcomp3[indexcomp - 1].append(50)
#                     resultcomp3[indexcomp - 1].append(50)
#                     resultcomp3[indexcomp - 1].append(50)
#
#                     sn = np.zeros((1,7));
#                     h = np.zeros((1,7)); # H(x)
#                     hp = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]);
#                     snp = rd;
#
#                     sn[0] = snp[1]; # Sn(x)
#                     h[0] = hp[0];
#
#                     ind1 = 1;
#                     ind2 = 1;
#
#                     for p1 in range (1,7):
#                         sn[p1] = sn[p1-1] + snp[0, p1];
#                         h[p1] = h[p1-1] + (hp[p1]);
#
#                     nsample = visreal;
#
#                     temp = np.abs(sn - h);
#                     temp2 = sn - h;
#                     temp3 = h - sn;
#
#                     d = np.max(temp);
#                     dminus = np.max(temp3);
#                     dplus = np.max(temp2);
#                     dminus = d;
#                     dplus = d;
#
#                     limitminus = nsample*(1 - dminus);
#                     limitplus = nsample*(1 - dplus);
#
#                     carrminus = [];
#                     barrminus = [];
#                     carrplus = [];
#                     barrplus = [];
#
#                     # calculating c values for d -
#                     for j in range(limitminus):
#                         tempvalminus = dminus + (j/nsample);
#                         okcmin = 0;
#                         for tempi in range(7):
#                             if (h[tempi] >= tempvalminus):
#                                 if (okcmin == 0):
#                                     okcmin = 1
#                                     carrminus.append(1 - h[tempi])
#
#                         if (okcmin == 0):
#                             carrminus.append(1 - tempvalminus);
#
#                     # calculating b values for d -
#                     for j in range (limitminus):
#                         barrminus.append(0);
#                         if (carrminus[j] > 0):
#                             barrminus[j] = 1
#                             for tempcount in range (j):
#                                 barrminus[j] = barrminus[j] - (sci.comb(j, tempcount) * (pow(carrminus[tempcount],(j - tempcount))) * barrminus[tempcount]);
#
#                     # calculating critical value of d -
#                     cvminus = 0;
#                     for j in range(limitminus):
#                         cvminus = cvminus + (sci.comb(nsample, j) * (pow(carrminus[j],(nsample - j))) * barrminus[j]);
#
#
#                     # calculating c values(f) for d +
#                     for j in range(limitplus):
#                         tempvalplus = 1 - dplus - (j/nsample)
#                         okcplus = 0;
#
#                         for tempi in range(7):
#                             if (h[tempi] <= tempvalplus):
#                                 carrplus.append(h[tempi])
#                                 okcplus = 1
#
#                         if (okcplus == 0):
#                             carrplus.append(tempvalplus);
#
#                     # calculating b values(e) for d +
#                     for j in range(limitplus):
#                         barrplus.append(0)
#                         if (carrplus[j] > 0):
#                             barrplus[j] = 1
#                             for tempcount in range(j-1):
#                                 barrplus[j] = barrplus[j] - (sci.comb(j, tempcount) * (pow(carrplus[tempcount],(j - tempcount))) * barrplus[tempcount])
#
#
#                     # calculating critical value of d +
#                     cvplus = 0;
#                     for j in range(limitplus):
#                         cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j],(nsample - j))) * barrplus[j])
#
#
#                     cv = cvplus + cvminus;
#                     # result = pdsum + pdminus;
#                     if (cv > 0.05):
#                         sssd = sssd + 1
#
#                     resultcomp3[indexcomp-1][5] = cv
#
#                 if ((vis1 > 0)):
#                     hp = 0;
#                     snp = 0;
#                     sn = 0;
#                     h = 0; % H(x)
#                     hp = pol1;
#                     snp = rd;
#
#                     sn(1) = snp(1); % Sn(x)
#                     h(1) = hp(1);
#
#                     ind1 = 2;
#                     ind2 = 2;
#
#                     for p1 = 2:1: 7
#                     sn(p1) = sn(p1 - 1) + snp(p1);
#                     h(p1) = h(p1 - 1) + (hp(p1));
#                 end
#
# % DISCRETE
# KOLMOGOROV
# nsample = visreal;
#
# temp = abs(sn - h);
# temp2 = sn - h;
# temp3 = h - sn;
#
# d = max(temp);
# dminus = max(temp3);
# dplus = max(temp2);
# dminus = d;
# dplus = d;
#
# limitminus = nsample * (1 - dminus);
# limitplus = nsample * (1 - dplus);
#
# carrminus = 0;
# barrminus = 0;
# carrplus = 0;
# barrplus = 0;
#
# % calculating
# c
# values
# for d -
#     for j = 0:1: limitminus
# tempvalminus = dminus + (j / nsample);
# okcmin = 0;
# for tempi = 1:1: 7
# if (h(tempi) >= tempvalminus)
#     if (okcmin == 0)
#         okcmin = 1;
#         carrminus(j + 1) = 1 - h(tempi);
#     end
# end
# end
#
# if (okcmin == 0)
#     carrminus(j + 1) = 1 - tempvalminus;
# end
# end
#
# % calculating
# b
# values
# for d -
#     for j = 0:1: limitminus
# barrminus(j + 1) = 0;
# if (carrminus(j + 1) > 0)
#     barrminus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrminus(j + 1) = barrminus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrminus(tempcount + 1) ^ (j - tempcount)) * barrminus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d -
# cvminus = 0;
# for j = 0:1: limitminus
# cvminus = cvminus + (nchoosek(nsample, j) * (carrminus(j + 1) ^ (nsample - j)) * barrminus(j + 1));
# end
#
# % calculating
# c
# values(f)
# for d +
#     for j = 0:1: limitplus
# tempvalplus = 1 - dplus - (j / nsample);
# okcplus = 0;
# for tempi = 1:1: 7
# if (h(tempi) <= tempvalplus)
#     carrplus(j + 1) = h(tempi);
# end
# end
#
# if (okcplus == 0)
#     carrplus(j + 1) = tempvalplus;
# end
# end
#
# % calculating
# b
# values(e)
# for d +
#     for j = 0:1: limitplus
# barrplus(j + 1) = 0;
# if (carrplus(j + 1) > 0)
#     barrplus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrplus(j + 1) = barrplus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrplus(tempcount + 1) ^ (j - tempcount)) * barrplus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d +
# cvplus = 0;
# for j = 0:1: limitplus
# cvplus = cvplus + (nchoosek(nsample, j) * (carrplus(j + 1) ^ (nsample - j)) * barrplus(j + 1));
# end
#
# cv = cvplus + cvminus;
# % result = pdsum + pdminus;
# if (cv > 0.05)
#     sss1 = sss1 + 1;
# end
#
# resultcomp3(indexcomp, 3) = cv;
# cnt1 = cnt1 + 1;
# visited1 = visited1 + 1;
# end
# if ((vis2 > 0))
#     hp = 0;
#     snp = 0;
#     sn = 0;
#     h = 0; % H(x)
#     hp = pol2;
#     snp = rd;
#
#     sn(1) = snp(1); % Sn(x)
#     h(1) = hp(1);
#
#     ind1 = 2;
#     ind2 = 2;
#
#     for p1 = 2:1: 7
#     sn(p1) = sn(p1 - 1) + snp(p1);
#     h(p1) = h(p1 - 1) + (hp(p1));
# end
#
# % DISCRETE
# KOLMOGOROV
# nsample = visreal;
#
# temp = abs(sn - h);
# temp2 = sn - h;
# temp3 = h - sn;
#
# d = max(temp);
# dminus = max(temp3);
# dplus = max(temp2);
# dminus = d;
# dplus = d;
#
# limitminus = nsample * (1 - dminus);
# limitplus = nsample * (1 - dplus);
#
# carrminus = 0;
# barrminus = 0;
# carrplus = 0;
# barrplus = 0;
#
# % calculating
# c
# values
# for d -
#     for j = 0:1: limitminus
# tempvalminus = dminus + (j / nsample);
# okcmin = 0;
# for tempi = 1:1: 7
# if (h(tempi) >= tempvalminus)
#     if (okcmin == 0)
#         okcmin = 1;
#         carrminus(j + 1) = 1 - h(tempi);
#     end
# end
# end
#
# if (okcmin == 0)
#     carrminus(j + 1) = 1 - tempvalminus;
# end
# end
#
# % calculating
# b
# values
# for d -
#     for j = 0:1: limitminus
# barrminus(j + 1) = 0;
# if (carrminus(j + 1) > 0)
#     barrminus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrminus(j + 1) = barrminus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrminus(tempcount + 1) ^ (j - tempcount)) * barrminus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d -
# cvminus = 0;
# for j = 0:1: limitminus
# cvminus = cvminus + (nchoosek(nsample, j) * (carrminus(j + 1) ^ (nsample - j)) * barrminus(j + 1));
# end
#
# % calculating
# c
# values(f)
# for d +
#     for j = 0:1: limitplus
# tempvalplus = 1 - dplus - (j / nsample);
# okcplus = 0;
# for tempi = 1:1: 7
# if (h(tempi) <= tempvalplus)
#     carrplus(j + 1) = h(tempi);
# end
# end
#
# if (okcplus == 0)
#     carrplus(j + 1) = tempvalplus;
# end
# end
#
# % calculating
# b
# values(e)
# for d +
#     for j = 0:1: limitplus
# barrplus(j + 1) = 0;
# if (carrplus(j + 1) > 0)
#     barrplus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrplus(j + 1) = barrplus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrplus(tempcount + 1) ^ (j - tempcount)) * barrplus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d +
# cvplus = 0;
# for j = 0:1: limitplus
# cvplus = cvplus + (nchoosek(nsample, j) * (carrplus(j + 1) ^ (nsample - j)) * barrplus(j + 1));
# end
#
# cv = cvplus + cvminus;
# % result = pdsum + pdminus;
# if (cv > 0.05)
#     sss2 = sss2 + 1;
# end
#
# resultcomp3(indexcomp, 4) = cv;
# cnt2 = cnt2 + 1;
# visited2 = visited2 + 1;
# end
# if ((vis3 > 0))
#     hp = 0;
#     snp = 0;
#     sn = 0;
#     h = 0; % H(x)
#     hp = pol3;
#     snp = rd;
#
#     sn(1) = snp(1); % Sn(x)
#     h(1) = hp(1);
#
#     ind1 = 2;
#     ind2 = 2;
#
#     for p1 = 2:1: 7
#     sn(p1) = sn(p1 - 1) + snp(p1);
#     h(p1) = h(p1 - 1) + (hp(p1));
# end
#
# % DISCRETE
# KOLMOGOROV
# nsample = visreal;
#
# temp = abs(sn - h);
# temp2 = sn - h;
# temp3 = h - sn;
#
# d = max(temp);
# dminus = max(temp3);
# dplus = max(temp2);
# dminus = d;
# dplus = d;
#
# limitminus = nsample * (1 - dminus);
# limitplus = nsample * (1 - dplus);
#
# carrminus = 0;
# barrminus = 0;
# carrplus = 0;
# barrplus = 0;
#
# % calculating
# c
# values
# for d -
#     for j = 0:1: limitminus
# tempvalminus = dminus + (j / nsample);
# okcmin = 0;
# for tempi = 1:1: 7
# if (h(tempi) >= tempvalminus)
#     if (okcmin == 0)
#         okcmin = 1;
#         carrminus(j + 1) = 1 - h(tempi);
#     end
# end
# end
#
# if (okcmin == 0)
#     carrminus(j + 1) = 1 - tempvalminus;
# end
# end
#
# % calculating
# b
# values
# for d -
#     for j = 0:1: limitminus
# barrminus(j + 1) = 0;
# if (carrminus(j + 1) > 0)
#     barrminus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrminus(j + 1) = barrminus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrminus(tempcount + 1) ^ (j - tempcount)) * barrminus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d -
# cvminus = 0;
# for j = 0:1: limitminus
# cvminus = cvminus + (nchoosek(nsample, j) * (carrminus(j + 1) ^ (nsample - j)) * barrminus(j + 1));
# end
#
# % calculating
# c
# values(f)
# for d +
#     for j = 0:1: limitplus
# tempvalplus = 1 - dplus - (j / nsample);
# okcplus = 0;
# for tempi = 1:1: 7
# if (h(tempi) <= tempvalplus)
#     carrplus(j + 1) = h(tempi);
# end
# end
#
# if (okcplus == 0)
#     carrplus(j + 1) = tempvalplus;
# end
# end
#
# % calculating
# b
# values(e)
# for d +
#     for j = 0:1: limitplus
# barrplus(j + 1) = 0;
# if (carrplus(j + 1) > 0)
#     barrplus(j + 1) = 1;
#     for tempcount = 0:1: (j - 1)
#     barrplus(j + 1) = barrplus(j + 1) - (
#                 nchoosek(j, tempcount) * (carrplus(tempcount + 1) ^ (j - tempcount)) * barrplus(tempcount + 1));
# end
# end
# end
#
# % calculating
# critical
# value
# of
# d +
# cvplus = 0;
# for j = 0:1: limitplus
# cvplus = cvplus + (nchoosek(nsample, j) * (carrplus(j + 1) ^ (nsample - j)) * barrplus(j + 1));
# end
#
# cv = cvplus + cvminus;
# % result = pdsum + pdminus;
# if (cv > 0.05)
#     sss3 = sss3 + 1;
# end
#
# resultcomp3(indexcomp, 5) = cv;
# cnt3 = cnt3 + 1;
# visited3 = visited3 + 1;
# end
#
# if ((vis1 > 0) | | (vis2 > 0) | | (vis3 > 0))
#     visited = visited + 1;
#     cnt = cnt + 1;
#     indexcomp = indexcomp + 1;
#     check = 1;
# end
# end
# end
# else
# sss1 = sss1 / cnt1;
# sss2 = sss2 / cnt2;
# sss3 = sss3 / cnt3;
# sssd = sssd / cnt;
#
# resultcomp2(index, 1) = currentcar;
# resultcomp2(index, 2) = sss1;
# resultcomp2(index, 3) = sss2;
# resultcomp2(index, 4) = sss3;
# resultcomp2(index, 5) = sssd;
# index = index + 1;
#
# chisq1 = 0;
# chisq2 = 0;
# chisq3 = 0;
#
# currentcar = realdata(i);
# i = i - 1;
#
# cnt = 0;
# cnt1 = 0;
# cnt2 = 0;
# cnt3 = 0;
#
# sss1 = 0;
# sss2 = 0;
# sss3 = 0;
# sssd = 0;
# check = 0;
# end
# i = i + 1;
# end

else:
    #file = open('level0/level0.txt', 'w')
    # level0pol = []
    #
    # for i in range(0, pow(3, Params.Params.numobservations - 1)):
    #     for j in range(0, Params.Params.numlanes):
    #         #file.write(str(i*5+j)+'\t')
    #         msg = int_to_msg(i,j)
    #         if msg.fc_v == 'approaching' and msg.fc_d == 'close':
    #            # file.write('4')
    #             level0pol.append(4)
    #         elif (msg.fc_v == 'approaching' and msg.fc_d == 'nominal') or (
    #                 msg.fc_v == 'stable' and msg.fc_d == 'close'):
    #             #file.write('2')
    #             level0pol.append(2)
    #         elif (msg.fc_v == 'retreating' and msg.fc_d == 'nominal') or (msg.fc_d == 'far'):
    #             #file.write('1')
    #             level0pol.append(1)
    #         else:
    #             #file.write('0')
    #             level0pol.append(0)
    #         #file.write('\n')
    #         print(str(5*i+j))
    # #file.close()

    with open('./data/realdata2.txt') as f:
        realdata = [[float(x) for x in line.split()] for line in f]

    currentcar = 2
    check = 0
    b = []
    bindex = 0
    totcar = 0
    index = 0
    resultcomp21 = []
    resultcomp22 = []
    alltotcar = 0
    for i in range(71401):
        if int(realdata[i][0]) == currentcar:
                check = 1
                sum = 0
                lane = int(realdata[i][1] % 5)
                num = int(realdata[i][1] / 5)
                msg = int_to_msg(num, lane)
                if msg.fc_v == 'approaching' and msg.fc_d == 'close':
                    actionget = 4
                elif (msg.fc_v == 'approaching' and msg.fc_d == 'nominal') or (
                                msg.fc_v == 'stable' and msg.fc_d == 'close'):
                    actionget = 2
                elif (msg.fc_v == 'retreating' and msg.fc_d == 'nominal') or (msg.fc_d == 'far'):
                    actionget = 1
                else:
                    actionget = 0

                acts = [0, 0, 0, 0, 0, 0, 0]
                acts[actionget] = 1
                realacts = [realdata[i][2], realdata[i][3], realdata[i][4], realdata[i][5], realdata[i][6],
                            realdata[i][7], realdata[i][8]]
                visreal = realdata[i][9]
                vispol = visreal
                K1 = math.sqrt(vispol / visreal)
                K2 = math.sqrt(visreal / vispol)
                for t in range(7):
                    if (vispol * acts[t] + visreal * realacts[t]) > 0:
                        sum += (math.pow((K1 * visreal * realacts[t] + K2 * vispol * acts[t]), 2) / (
                        vispol * acts[t] + visreal * realacts[t]))
                # if actionget == realdata[i][2]:
                #     b.append(1)
                # else:
                #     b.append(0)
                b.append(sum)
                bindex += 1
        else:
            if check == 1:
                tempsum = 0
                totcar += 1
                resultcomp21.append(currentcar)
                for h in range(len(b)):
                    # if b[h] == 1:
                    #     tempsum+=1
                    if b[h] < 14.7:
                        tempsum += 1

                resultcomp22.append(tempsum / bindex)
                # for h in range(len(b)):
                #     if b[h] == 1:
                #         tempsum+=1
                # a = tempsum/bindex
                # resultcomp22.append(a)
            else:
                resultcomp21.append(currentcar)
                resultcomp22.append(-1)
            index += 1
            bindex = 0
            b = []
            check = 0
            currentcar = int(realdata[i][0])
            alltotcar += 1
            i -= 1
    carmod = 0;
    level0cars = []
    for i in range(len(resultcomp22)):
        if resultcomp22[i] >= 0.7:
            carmod += 1
            level0cars.append(resultcomp21[i])
    print(100 * carmod / 2168)

    file = open('level0/resultcomp.txt', 'w')
    for q in range(2168):
        file.write(str(resultcomp21[q]) + '\t' + str(resultcomp22[q]))
        file.write('\n')
    file.close()

    file = open('level0/level0cars.txt', 'w')
    for q in range(len(level0cars)):
        file.write(str(level0cars[q]))
        file.write('\n')
    file.close()














