import Params, Message, DQNAgent, State
import numpy as np
from numpy import array
import math
import scipy.special as sci

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

filter = 2

print('starting comp')
numsample = 100

filter_nums = [2,4]
state_size = 19
action_size = 7
agentl1 = DQNAgent.DQNAgent(state_size, action_size)
agentl2 = DQNAgent.DQNAgent(state_size, action_size)
agentl3 = DQNAgent.DQNAgent(state_size, action_size)
agentl1.load("datanew/level1/weight43.h5")
agentl2.load("datanew/level2/weight49.h5")
agentl3.load("datanew/weight43.h5")
agentl1.T = 1
agentl2.T = 1
agentl3.T = 1

with open('data/realdata.txt') as f:
    realdata = [[float(x) for x in line.split()] for line in f]

for filter_count in range(len(filter_nums)):
    filter = filter_nums[filter_count]
    print('Filter Num:'+str(filter))

    mae_accept = 0
    mae_reject = 0
    mae_accept_count = 0
    mae_reject_count = 0

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
            if (realdata[i][10] > filter):
                # print(i)
                allvisited = allvisited + 1;
                crtval = 12.59;
                rd = 0;
                pol = 0;
                pol1 = 0;
                pol2 = 0;
                pol3 = 0;
                dumb = np.array([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]);
                vis1 = 100  # visits1[int(realdata[i][1])]
                vis2 = 100  # visits2[int(realdata[i][1])]
                vis3 = 100  # visits3[int(realdata[i][1])]
                visreal = realdata[i][10]

                lane = int(realdata[i][1] % 5)
                num = int(realdata[i][1] / 5)
                numl = lane
                msg = int_to_msg(num, lane)
                pol1 = pol2 = pol3 = 0
                statelist = []
                for tempvar in range(numsample):
                    currentstate = None
                    currentstate = state.get_contMessageRandom(state.get_intMessage3(msg),numl)
                    currentstate = np.reshape(currentstate, [1, 1, state_size])
                    statelist.append(currentstate)
                statelist = np.reshape(np.asarray(statelist),[numsample,1,state_size])
                pol1 = np.mean(agentl1.getq_prob_batch(statelist),axis=0).reshape((1,7))
                pol2 = np.mean(agentl2.getq_prob_batch(statelist),axis=0).reshape((1,7))
                pol3 = np.mean(agentl3.getq_prob_batch(statelist),axis=0).reshape((1,7))
                rd = np.zeros((1, 7))

                p1ind = []
                p2ind = []
                p3ind = []
                rdind = []
                p1sum = 1
                p2sum = 1
                p3sum = 1
                rdsum = 1
                for p1 in range(7):
                    if (pol1[0, p1] < 0.01):
                        p1sum += 0.01 - pol1[0, p1]
                        pol1[0, p1] = 0.01
                    if (pol2[0, p1] < 0.01):
                        p2sum += 0.01 - pol2[0, p1]
                        pol2[0, p1] = 0.01
                    if (pol3[0, p1] < 0.01):
                        p3sum += 0.01 - pol3[0, p1]
                        pol3[0, p1] = 0.01

                    rd[0, p1] = realdata[i][p1 + 2] / realdata[i][10]
                    if (rd[0, p1] < 0.01):
                        rdsum += 0.01 - rd[0, p1]
                        rd[0, p1] = 0.01

                cntind = 0
                for cntind in range(7):
                    pol1[0, cntind] /= p1sum
                    pol2[0, cntind] /= p2sum
                    pol3[0, cntind] /= p3sum
                    rd[0, cntind] /= rdsum

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

                    sn = np.zeros((1, 7));
                    h = np.zeros((1, 7));  # H(x)
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
                    if limitminus<0:
                        ggg = 0
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

                    resultcomp3[indexcomp - 1][5] = cv

                if ((vis1 > 0)):
                    dtemp = 0;
                    sn = np.zeros((1, 7));
                    h = np.zeros((1, 7));  # H(x)
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
                    if limitminus<0:
                        ggg = 0
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
                        mae_accept_count += 1
                        mae_accept += np.sum(np.abs(hp - snp))
                    else:
                        mae_reject_count += 1
                        mae_reject += np.sum(np.abs(hp - snp))

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
                    if limitminus<0:
                        ggg = 0
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
                        mae_accept_count += 1
                        mae_accept += np.sum(np.abs(hp - snp))
                    else:
                        mae_reject_count += 1
                        mae_reject += np.sum(np.abs(hp - snp))

                    resultcomp3[indexcomp - 1][3] = cv
                    cnt2 = cnt2 + 1;
                    visited2 = visited2 + 1;

                if ((vis3 > 0)):
                    hdtemp = 0;
                    sn = np.zeros((1, 7));
                    h = np.zeros((1, 7));  # H(x)
                    hp = pol3
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
                    if limitminus<0:
                        ggg = 0
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
                                barrplus[j] -= (
                                            sci.comb(j, tempcount) * (pow(carrplus[tempcount], (j - tempcount))) * barrplus[
                                        tempcount])

                    # calculating critical value of d +
                    cvplus = 0;
                    for j in range(int(limitplus)):
                        cvplus = cvplus + (sci.comb(nsample, j) * (pow(carrplus[j], (nsample - j))) * barrplus[j])

                    cv = cvplus + cvminus;
                    # result = pdsum + pdminus;
                    if (cv > 0.05):
                        sss3 = sss3 + 1
                        mae_accept_count += 1
                        mae_accept += np.sum(np.abs(hp - snp))
                    else:
                        mae_reject_count += 1
                        mae_reject += np.sum(np.abs(hp - snp))

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

    acpt = mae_accept / mae_accept_count
    rjct = mae_reject / mae_reject_count
    np.savetxt('results_final/' + str(filter + 1) + '/accmae.txt', np.reshape(np.asarray(acpt), (1, 1)), delimiter='\t')
    np.savetxt('results_final/' + str(filter + 1) + '/rejmae.txt', np.reshape(np.asarray(rjct), (1, 1)), delimiter='\t')
    rescomp2tmp = np.asarray(resultcomp2)
    rescomp3tmp = np.asarray(resultcomp3)
    np.savetxt('results_final/'+str(filter+1)+'/resultcomp2.txt', rescomp2tmp, delimiter='\t')
    np.savetxt('results_final/'+str(filter+1)+'/resultcomp3.txt', rescomp3tmp, delimiter='\t')

