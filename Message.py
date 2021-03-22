import Params

class Message:

    def __init__(self, msg):
        self.msg = msg
        self.fll_d = msg[0]
        self.fll_v = msg[1]
        self.frr_d = msg[2]
        self.frr_v = msg[3]
        self.rll_d = msg[4]
        self.rll_v = msg[5]
        self.rrr_d = msg[6]
        self.rrr_v = msg[7]
        self.fl_d = msg[8]
        self.fl_v = msg[9]
        self.fc_d = msg[10]
        self.fc_v = msg[11]
        self.fr_d = msg[12]
        self.fr_v = msg[13]
        self.rl_d = msg[14]
        self.rl_v = msg[15]
        self.rr_d = msg[16]
        self.rr_v = msg[17]
        self.lane = msg[18]

    def equals(self, test):
        for i in range (0, Params.Params.numobservations):
            if test.msg[i] != self.msg[i]:
                return False
        return True

    def print_msg(self):
        print('--------------------------', end='\n')
        print('Message', end='\n')
        print("--------------------------", end='\n')
        print(self.rl_d+"\t\t\t"+self.fl_d, end='\n')
        print(self.rl_v+"\t\t\t"+self.fl_v, end='\n')
        print("--  --  --  --  --  --  --", end='\n')
        print("\t\t|This|\t\t" + self.fc_d, end='\n')
        print("\t\t|Car |\t\t" + self.fc_v, end='\n')
        print("--  --  --  --  --  --  --", end='\n')
        print(self.rr_d+"\t\t\t"+self.fr_d, end='\n')
        print(self.rr_v+"\t\t\t"+self.fr_v, end='\n')
        print("--------------------------", end='\n')
        print("Lane: " + self.lane, end='\n')










