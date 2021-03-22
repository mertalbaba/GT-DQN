import Params, Message
import math
class MApairs:
    def __init__(self, msg):
        self.msg = msg
        self.msg_count = 0
        self.msg_value_V = 0
        self.msg_beta = 0
        self.act_prob = [1/(Params.Params.numactions) for i in range(Params.Params.numactions)]
        self.act_value_Q = [0 for i in range(Params.Params.numactions)]
        self.act_count = [0 for i in range(Params.Params.numactions)]
        self.act_beta = [0 for i in range(Params.Params.numactions)]
        self.total_actcount = 0
        self.entropy = 2.8

    def increase_prob(self, act_index):
        increment = 0.01
        self.act_prob[act_index] += increment
        for i in range (0, Params.Params.numactions):
            self.act_prob[i] /= (1+increment)

    def increase_countact(self, act_index):
        self.act_count[act_index] += 1
        self.total_actcount += 1

    def update_entropy(self):
        self.entropy = 0
        for i in range (0, Params.Params.numactions):
            if self.act_prob[i] != 0:
                self.entropy += ((self.act_prob[i])*(math.log2(1/self.act_prob[i])));

        if self.entropy == 0:
            self.entropy = 2.8

    def set_probs(self, probs):
        if len(probs) != Params.Params.numactions:
            print('Problem with input array size in set_probs().', end='\n')

        sum = 0
        for i in range (0, Params.Params.numactions):
            self.act_prob[i] = probs[i]
            sum += probs[i]

        for i in range (0, Params.Params.numactions):
            self.act_prob[i] /= sum
