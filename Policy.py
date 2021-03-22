import Action, Params, MApairs, Message, scanner
import random

class Policy:
    msg_list = []
    act_list = []

#    def __init__(self):
    #        self.add_all_actiobs()
    #        for i in range (0, 3**(Params.Params.numobservations-1)):
    #           for j in range(0,5):
    #              self.msg_list.append(MApairs.MApairs(self.int_to_msg(i, j)))

    def __init__(self, str):
        self.add_all_actiobs()
        if(str == 'level 0'):
            self.build_level_0()
        elif(str == 'Static'):
            self.build_Static()
        else:
            sc = scanner.Scanner(str)
            while sc.has_next():
                temp = sc.next_int()
                newMA = MApairs.MApairs(self.int_to_msg(int(temp/5),(temp%5)))
                newMA.msg_value_V = sc.next_float()
                newMA.msg_count = sc.next_float()
                newMA.msg_beta = sc.next_float()
                for i in range(0, Params.Params.numactions):
                    newMA.act_prob[i] = sc.next_float()
                    newMA.act_value_Q[i] = sc.next_float()
                    newMA.act_count[i] = sc.next_float()
                    newMA.act_beta[i] = sc.next_float()

                self.msg_list.append(newMA)

    def build_level_0(self):
        always_decelerate = [0,0,1,0,0,0,0]
        always_hard_decelerate = [0,0,0,0,1,0,0]
        always_maintain = [1,0,0,0,0,0,0]
        always_accelerate = [0,1,0,0,0,0,0]
        possible_decelerate = [0,0,1,0,0]
        possible_accelerate = [0,1,0,0,0]

        for i in range (0, pow(3, Params.Params.numobservations-1)):
            for j in range (0, Params.Params.numlanes):
                newMA = MApairs.MApairs(self.int_to_msg(i,j))
                if newMA.msg.fc_v == 'approaching' and newMA.msg.fc_d == 'close':
                    newMA.set_probs(always_hard_decelerate)
                elif (newMA.msg.fc_v == 'approaching' and newMA.msg.fc_d == 'nominal') or (newMA.msg.fc_v == 'stable' and newMA.msg.fc_d == 'close'):
                    newMA.set_probs(always_decelerate)
                elif (newMA.msg.fc_v == 'retreating' and newMA.msg.fc_d == 'nominal') or (newMA.msg.fc_d == 'far'):
                    newMA.set_probs(always_accelerate)
                else:
                    newMA.set_probs(always_maintain)
                self.msg_list.append(newMA)

    def build_Static(self):
        always_maintain = [1, 0, 0, 0, 0, 0, 0]
        for i in range (0, pow(3, Params.Params.numobservations-1)):
            for j in range (0, Params.Params.numlanes):
                newMA = MApairs.MApairs(self.int_to_msg(i,j))
                newMA.set_probs(always_maintain)
                self.msg_list.append(newMA)

    def get_Action(self, msg):
        index = ''
        tempmultip = 0
        tempnum = 0
        for i in range(0,18,2):
            index = index + self.d_char(msg, i)
            index = index + self.v_char(msg, i+1)
        if self.lane_char(msg,18)=='*':
            tempmultip = 1
        else:
            tempmultip = 1+ord(self.lane_char(msg,18))-ord('0')

        ma = self.msg_list[5*str2int(index,3)+tempmultip-1]
        total = 0
        rand = random.uniform(0.0, 1.0)
        for i in range (0, len(self.act_list)):
            total += ma.act_prob[i]
            if rand<total:
                return self.act_list[i]

        print('Error in get_Action()')
        return None



    def add_all_actiobs(self):
        self.act_list.append(Action.maintain)
        self.act_list.append(Action.accelerate)
        self.act_list.append(Action.decelerate)
        self.act_list.append(Action.hard_accelerate)
        self.act_list.append(Action.hard_decelerate)
        self.act_list.append(Action.moveleft)
        self.act_list.append(Action.moveright)

    def int_to_msg(self, num, lane):
        base3 = self.int2base(num, 3)
        base3 = '{:18s}'.format(base3)
        j = base3.replace(' ', '0')
        base3 = j
        msgwords = []
        for i in range(0,18,2):
            msgwords.append(self.d_word(base3[i]))
            msgwords.append(self.v_word(base3[i]))
        msgwords.append(self.lane_word(str(lane)))
        return Message.Message(msgwords)


    def d_word(self, i):
        if(i == '0'):
            return 'close'
        elif (i =='1'):
            return 'nominal'
        elif (i == '2'):
            return 'far'
        else:
            print('cant find '+i+' in d_word() options')
            return '*'

    def v_word(self, i):
        if(i == '0'):
            return 'stable'
        elif (i =='1'):
            return 'approaching'
        elif (i == '2'):
            return 'retreating'
        else:
            print('cant find '+i+' in v_word() options')
            return '*'

    def lane_word(self, i):
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

    def d_char(self, msg, i):
        if(msg.msg[i] == 'close'):
            return '0'
        elif (msg.msg[i] =='nominal'):
            return '1'
        elif (msg.msg[i] == 'far'):
            return '2'
        else:
            print('cant find '+msg.msg[i]+' in d_char() options')
            return '*'

    def v_char(self, msg, i):
        if(msg.msg[i] == 'stable'):
            return '0'
        elif (msg.msg[i] =='approaching'):
            return '1'
        elif (msg.msg[i] == 'retreating'):
            return '2'
        else:
            print('cant find '+msg.msg[i]+' in v_char() options')
            return '*'

    def lane_char(self, msg, i):
        if(msg.msg[i] == 'right'):
            return '0'
        elif (msg.msg[i] =='center1'):
            return '1'
        elif (msg.msg[i] == 'center2'):
            return '2'
        elif (msg.msg[i] == 'center3'):
            return '3'
        elif (msg.msg[i] == 'left'):
            return '4'
        else:
            print('cant find '+msg.msg[i]+' in lane_char() options')
            return '*'

    def save_policy(self, filename, entropy_episode, collusion, entropy_states, control):
        average = 0
        visited_average = 0
        total_entropy = 0
        entropy_count = 0
        visited_total = 0
        visited_count = 0

        file = open(filename, 'w')
        file2 = open('data/collusion.dat','w')
        file3 = open('data/entropy.dat','w')
        file4 = open('data/entropy_states.dat','w')
        file.truncate(0)
        file2.truncate(0)
        file3.truncate(0)
        file4.truncate(0)

        print('Saving policy to file:'+filename)

        index = 0
        for ma in self.msg_list:
            if ma.entropy == None:
                total_entropy += 2.8
            else:
                total_entropy+=ma.entropy
                if ma.entropy != 1:
                    visited_total += ma.entropy
                    visited_count += 1

            entropy_count += 1
            strr = ''+str(index)+'\t'+str(ma.msg_value_V)+'\t'+str(ma.msg_count)+'\t'+str(ma.msg_beta)+'\t'
            for i in range (0, Params.Params.numactions):
                strr += str(ma.act_prob[i])+'\t'+str(ma.act_value_Q[i])+'\t'+str(ma.act_count[i])+'\t'+str(ma.act_beta[i])+'\t'
            file.write(strr+'\n')
            index += 1
        file.close()

        for k in range (0,1):
            file2.write(str(collusion[k]))
            file3.write(str(entropy_episode[k]))
        file2.close()
        file3.close()

#        if control:
 #           for i in range (0,6000):
  #              if(entropy_states[i] != 0.0):
   #                 file4.write(entropy_states[i])
    #    file4.close()




    def int2base(self, x, b, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
        'convert an integer to its string representation in a given base'
        if b < 2 or b > len(alphabet):
            if b == 64:  # assume base64 rather than raise error
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
            else:
                raise AssertionError("int2base base out of range")
        if isinstance(x, complex):  # return a tuple
            return (self.int2base(x.real, b, alphabet), self.int2base(x.imag, b, alphabet))
        if x <= 0:
            if x == 0:
                return alphabet[0]
            else:
                return '-' + self.int2base(-x, b, alphabet)
        # else x is non-negative real
        rets = ''
        while x > 0:
            x, idx = divmod(x, b)
            rets = alphabet[idx] + rets
        return rets

def str2int(x, base):
    temp = 0
    ind = len(x) - 1;
    for i in range(0, len(x)):
        temp += pow(base, ind)*int(x[i])
        ind -= 1
    return temp

#level0 = Policy('level 0')
#level1 = Policy('data/Level1_Policy.dat')
#level2 = Policy('data/Level2_Policy.dat')
#Static = Policy('Static')

