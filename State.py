import Car, Params, Action, Message
import random, types
import numpy as np
import random


class State:
    cars = []

    def __init__(self, *args):

        if len(args) == 1:
            foundOneArg = True
            theOnlyArg = args[0]
        elif len(args) == 2:
            foundOneArg = False

        if foundOneArg and isinstance(theOnlyArg, int):
            self.init1(theOnlyArg)
        elif foundOneArg and isinstance(theOnlyArg, State):
            self.init3(theOnlyArg)
        else:
            self.init2(args[0], args[1])

    def init1(self, numcars):
        self.cars = []
        self.numcars = numcars
        for i in range(0, numcars):
            if i == 0:
                pol = 0
            else:
                pol = 0
            self.cars.append(Car.Car(i, pol))
        self.select_car_positions()

    def init2(self, numcars, pol0):
        self.cars = []
        self.numcars = numcars
        for i in range(0, numcars):
            if i == 0:
                pol = 0
            else:
                pol = 0
            self.cars.append(Car.Car(i, pol))
        self.select_car_positions()

    def init3(self, copy):
        self.cars = []
        self.numcars = copy.numcars
        for i in range(0, self.numcars):
            self.cars.append(copy.cars[i])

    def select_car_positions(self):
        self.zero_out_positions()
        for i in range(1, self.numcars):
            position_x = 0
            lane = random.randint(0, 4)
            attempt_count = 0
            acceptable = False
            while not acceptable:
                position_x = Params.Params.init_size * (random.uniform(0.0, 1.0) - 0.5)
                for j in range(i):
                    if lane == self.cars[j].lane and abs(position_x - self.cars[j].position_x) < (
                            Params.Params.min_initial_separation + Params.Params.carlength):
                        acceptable = False
                        break
                    else:
                        acceptable = True
                attempt_count += 1
                if attempt_count > 250:
                    # print('unable to place car within lane successfully')
                    lane = random.randint(0, 4)
                    attempt_count = 0

            self.cars[i].position_x = position_x
            self.cars[i].lane = lane
            self.cars[i].position_y = lane * Params.Params.lanewidth + 0.5 * Params.Params.lanewidth

    # def select_car_positions(self):
    #     self.zero_out_positions()
    #     for i in range(1, self.numcars):
    #         position_x = 0
    #         lane = random.randint(0,4)
    #         attempt_count = 0
    #         acceptable = False
    #         while not acceptable:
    #             position_x = Params.Params.init_size*(random.uniform(0.0,1.0)-0.5)
    #             for j in range (i):
    #                 if lane == self.cars[j].lane and abs(position_x-self.cars[j].position_x)<(Params.Params.min_initial_separation+Params.Params.carlength):
    #                     acceptable = False
    #                     break
    #                 else:
    #                     acceptable = True
    #             attempt_count += 1
    #             if attempt_count >250:
    #                 #print('unable to place car within lane successfully')
    #                 lane = random.randint(0, 4)
    #                 attempt_count = 0
    #
    #         self.cars[i].position_x = position_x
    #         self.cars[i].lane = lane
    #         self.cars[i].position_y = lane*Params.Params.lanewidth+0.5*Params.Params.lanewidth

    def zero_out_positions(self):
        for temp in self.cars:
            temp.lane = random.randint(0, 4)
            temp.position_x = 0
            temp.position_y = temp.lane * Params.Params.lanewidth + 0.5 * Params.Params.lanewidth
            speed_randomness = random.uniform(0.0, 2.0)
            temp.velocity_x = 10 + 1 * speed_randomness
            temp.current_action = Action.maintain
            temp.effort = 0
            temp.lanechange_flag = False

    def assign_policy(self):
        return 0

    # def update(self, Switch, action):
    #     for temp in self.cars:
    #         if temp.equals(self.cars[0]):
    #             temp.updateMotionNew(self.get_Message(temp), self.cars[0], self.get_numericMessage(temp), Switch, action)
    #         else:
    #             curr = self.get_intMessage2(temp)
    #             curr = np.reshape(curr, [1, 1,19])
    #             actiongetted = LaneChange.agentlev1.model.predict(curr)
    #             actionl = np.argmax(actiongetted[0])
    #             temp.updateMotionLevel(self.get_Message(temp), self.cars[0], self.get_numericMessage(temp), Switch, actionl)

    def critical(self):
        critical_lon = Params.Params.critical_distance
        critical_lat = Params.Params.lanewidth
        for temp in self.cars:
            if temp.equals(self.cars[0]):
                continue
            else:
                if temp.position_x >= 0 and temp.position_x <= critical_lon and abs(
                        temp.position_y - self.cars.get(0).position_y) <= critical_lat:
                    return True
        return False

    def collision_checkandreset(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.001
            if abs(temp.position_x) <= (Params.Params.carlength + epsilon) and abs(
                    temp.position_y - car0.position_y) <= (Params.Params.carwidth + epsilon):
                # temp.position_x = Params.Params.init_size+200
                # temp.velocity_x = 10
                return True
        return False

    def collision_checkandreset3(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.1
            if abs(temp.position_x - car0.position_x) <= (Params.Params.carlength + epsilon) and temp.lane == car0.lane:
                temp.position_x = Params.Params.init_size + 200
                temp.velocity_x = 10
                return True
        return False

    def collision_checkandreset_new(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.1
            if (temp.prev_position_x-car0.prev_position_x > Params.Params.carlength and abs(temp.position_x-car0.position_x) <= (Params.Params.carlength + epsilon) and temp.lane==car0.lane) \
                    or (abs(temp.position_x-car0.position_x) <= (Params.Params.carlength + epsilon) and (abs(car0.prev_action.vy)+abs(car0.current_action.vy)>0)
                        and temp.lane==car0.lane):
                temp.position_x = -Params.Params.init_size-100
                temp.velocity_x = 5
                return True
        return False

    def collision_check_new(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.1
            if (temp.prev_position_x-car0.prev_position_x > 0 and abs(temp.position_x-car0.position_x) <= (Params.Params.carlength + epsilon) and temp.lane==car0.lane) \
                    or (abs(temp.position_x-car0.position_x) <= (Params.Params.carlength + epsilon) and (abs(car0.prev_action.vy)+abs(car0.current_action.vy)>0)
                        and temp.lane==car0.lane):
                return True
        return False


    def collision_checkandreset4(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.1
            if abs(temp.position_x - car0.position_x) <= (Params.Params.carlength + epsilon) and temp.lane == car0.lane:
                return True
        return False

    def collision_check_sim(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.1
            if temp.position_x - car0.position_x > 0 and abs(temp.position_x - car0.position_x) <= (
                    Params.Params.carlength + epsilon) and temp.lane == car0.lane:
                return True
        return False

    def collision_checkandreset2(self):
        for car0 in self.cars:
            for temp in self.cars:
                if (not temp.equals(car0)) and (car0.lane == temp.lane):
                    epsilon = 0.001
                    if abs(temp.position_x - car0.position_x) <= (Params.Params.carlength + epsilon) and abs(
                            temp.position_y - car0.position_y) <= (Params.Params.carwidth + epsilon):
                        # temp.position_x = 2*Params.Params.init_size+200
                        # temp.velocity_x = 5
                        return True
        return False

    def collision_check(self):
        car0 = self.cars[0]
        for temp in self.cars:
            if temp.equals(car0):
                if car0.effort == 4:
                    return True
                continue

            epsilon = 0.001
            if abs(temp.position_x) <= (Params.Params.carlength + epsilon) and abs(
                    temp.position_y - car0.position_y) <= (Params.Params.carwidth + epsilon):
                temp.position_x = Params.Params.init_size + 200
                temp.velocity_x = 10
                return True
        return False

    def get_Message(self, msgcar):
        fll_d = Params.Params.max_sight_distance
        fll_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        frr_d = Params.Params.max_sight_distance
        frr_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        rll_d = -Params.Params.max_sight_distance
        rll_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        rrr_d = -Params.Params.max_sight_distance
        rrr_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        fl_d = Params.Params.max_sight_distance
        fl_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        fc_d = Params.Params.max_sight_distance
        fc_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        fr_d = Params.Params.max_sight_distance
        fr_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        rl_d = -Params.Params.max_sight_distance
        rl_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        rr_d = -Params.Params.max_sight_distance
        rr_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        lane = msgcar.lane

        carl = Params.Params.carlength

        for temp in self.cars:
            if temp.equals(msgcar):
                continue

            rel_position = temp.position_x - msgcar.position_x
            rel_velocity = temp.velocity_x - msgcar.velocity_x
            if temp.lane == (msgcar.lane + 2):
                if rel_position < -carl and rel_position > rl_d:
                    rll_d = rel_position + carl
                    rll_v = rel_velocity
                elif rel_position < 0 and rel_position > rl_d:
                    rll_d = 0
                    rll_v = rel_velocity
                elif rel_position >= carl and rel_position < fl_d:
                    fll_d = rel_position - carl
                    fll_v = rel_velocity
                elif rel_position >= 0 and rel_position < fl_d:
                    fll_d = 0
                    fll_v = rel_velocity

            elif temp.lane == (msgcar.lane - 2):
                if rel_position < -carl and rel_position > rr_d:
                    rrr_d = rel_position + carl
                    rrr_v = rel_velocity
                elif rel_position < 0 and rel_position > rr_d:
                    rrr_d = 0
                    rrr_v = rel_velocity
                elif rel_position >= carl and rel_position < fr_d:
                    frr_d = rel_position - carl
                    frr_v = rel_velocity
                elif rel_position >= 0 and rel_position < fr_d:
                    frr_d = 0
                    frr_v = rel_velocity

            elif temp.lane == (msgcar.lane + 1):
                if rel_position < -carl and rel_position > rl_d:
                    rl_d = rel_position + carl
                    rl_v = rel_velocity
                elif rel_position < 0 and rel_position > rl_d:
                    rl_d = 0
                    rl_v = rel_velocity
                elif rel_position >= carl and rel_position < fl_d:
                    fl_d = rel_position - carl
                    fl_v = rel_velocity
                elif rel_position >= 0 and rel_position < fl_d:
                    fl_d = 0
                    fl_v = rel_velocity

            elif temp.lane == (msgcar.lane - 1):
                if rel_position < -carl and rel_position > rr_d:
                    rr_d = rel_position + carl
                    rr_v = rel_velocity
                elif rel_position < 0 and rel_position > rr_d:
                    rr_d = 0
                    rr_v = rel_velocity
                elif rel_position >= carl and rel_position < fr_d:
                    fr_d = rel_position - carl
                    fr_v = rel_velocity
                elif rel_position >= 0 and rel_position < fr_d:
                    fr_d = 0
                    fr_v = rel_velocity

            elif temp.lane == msgcar.lane:
                if rel_position >= carl and rel_position < fc_d:
                    fc_d = rel_position - carl
                    fc_v = rel_velocity
                elif rel_position >= 0 and rel_position < fc_d:
                    fc_d = 0
                    fc_v = rel_velocity

        msgcar.distf = fc_d

        msg = []
        msg.append(fll_d)
        msg.append(fll_v)
        msg.append(frr_d)
        msg.append(frr_v)

        msg.append(rll_d)
        msg.append(rll_v)
        msg.append(rrr_d)
        msg.append(rrr_v)

        msg.append(fl_d)
        msg.append(fl_v)
        msg.append(fc_d)
        msg.append(fc_v)
        msg.append(fr_d)
        msg.append(fr_v)

        msg.append(rl_d)
        msg.append(rl_v)
        msg.append(rr_d)
        msg.append(rr_v)

        msg.append(lane)

        return msg

    def get_numericMessage(self, msgcar):
        fll_d = Params.Params.max_sight_distance
        fll_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        frr_d = Params.Params.max_sight_distance
        frr_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        rll_d = -Params.Params.max_sight_distance
        rll_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        rrr_d = -Params.Params.max_sight_distance
        rrr_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        fl_d = Params.Params.max_sight_distance
        fl_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        fc_d = Params.Params.max_sight_distance
        fc_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        fr_d = Params.Params.max_sight_distance
        fr_v = Params.Params.max_speed - msgcar.velocity_x + 0.1
        rl_d = -Params.Params.max_sight_distance
        rl_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        rr_d = -Params.Params.max_sight_distance
        rr_v = (Params.Params.min_speed - msgcar.velocity_x) - 0.1
        lane = msgcar.lane

        for temp in self.cars:
            if temp.equals(msgcar):
                continue

            rel_position = temp.position_x - msgcar.position_x
            rel_velocity = temp.velocity_x - msgcar.velocity_x
            if temp.lane == (msgcar.lane + 2):
                if rel_position < 0 and rel_position > rl_d:
                    rll_d = rel_position
                    rll_v = rel_velocity
                elif rel_position >= 0 and rel_position < fl_d:
                    fll_d = rel_position
                    fll_v = rel_velocity
            elif temp.lane == (msgcar.lane - 2):
                if rel_position < 0 and rel_position > rr_d:
                    rrr_d = rel_position
                    rrr_v = rel_velocity
                elif rel_position >= 0 and rel_position < fr_d:
                    frr_d = rel_position
                    frr_v = rel_velocity
            elif temp.lane == (msgcar.lane + 1):
                if rel_position < 0 and rel_position > rl_d:
                    rl_d = rel_position
                    rl_v = rel_velocity
                elif rel_position >= 0 and rel_position < fl_d:
                    fl_d = rel_position
                    fl_v = rel_velocity
            elif temp.lane == (msgcar.lane - 1):
                if rel_position < 0 and rel_position > rr_d:
                    rr_d = rel_position
                    rr_v = rel_velocity
                elif rel_position >= 0 and rel_position < fr_d:
                    fr_d = rel_position
                    fr_v = rel_velocity
            elif temp.lane == msgcar.lane:
                if rel_position >= 0 and rel_position < fc_d:
                    fc_d = rel_position
                    fc_v = rel_velocity

        msg = []
        msg.append(fll_d)
        msg.append(fll_v)
        msg.append(frr_d)
        msg.append(frr_v)
        msg.append(rll_d)
        msg.append(rll_v)
        msg.append(rrr_d)
        msg.append(rrr_v)
        msg.append(fl_d)
        msg.append(fl_v)
        msg.append(fc_d)
        msg.append(fc_v)
        msg.append(fr_d)
        msg.append(fr_v)
        msg.append(rl_d)
        msg.append(rl_v)
        msg.append(rr_d)
        msg.append(rr_v)
        return msg

    def get_intMessage(self, msgcar):
        msg = self.get_Message(msgcar)
        index = ''
        tempmultip = 0
        tempnum = 0
        for i in range(0, 18, 2):
            index = index + self.d_char(msg, i)
            index = index + self.v_char(msg, i + 1)
        if self.lane_char(msg, 18) == '*':
            tempmultip = 1
        else:
            tempmultip = 1 + ord(self.lane_char(msg, 18)) - ord('0')

        res = 5 * self.str2int(index, 3) + tempmultip - 1
        return res

    def get_intMessage2(self, msgcar):
        msg = self.get_Message(msgcar)
        index = []
        for i in range(0, 18, 2):
            index.append(ord(self.d_char(msg, i)) - ord('0'))
            index.append(ord(self.v_char(msg, i + 1)) - ord('0'))
        index.append(ord(self.lane_char(msg, 18)) - ord('0'))
        return index

    def get_intMessage3(self, msg):
        index = []
        for i in range(0, 18, 2):
            index.append(ord(self.d_char(msg, i)) - ord('0'))
            index.append(ord(self.v_char(msg, i + 1)) - ord('0'))
        index.append(ord(self.lane_char(msg, 18)) - ord('0'))
        return index

    def get_contMessage(self,msg,lane):
        res = []
        for i in range(4):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(11)
                elif msg[i] == 1:
                    res.append(27)
                else:
                    res.append(40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(-1)
                else:
                    res.append(1)

        for i in range(4,8):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(-11)
                elif msg[i] == 1:
                    res.append(-27)
                else:
                    res.append(-40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(1)
                else:
                    res.append(-1)

        for i in range(8,14):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(11)
                elif msg[i] == 1:
                    res.append(27)
                else:
                    res.append(40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(-1)
                else:
                    res.append(1)

        for i in range(14,18):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(-11)
                elif msg[i] == 1:
                    res.append(-27)
                else:
                    res.append(-40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(1)
                else:
                    res.append(-1)

        res.append(lane)

        return res

    def get_contMessageRandom(self,msg,lane):
        res = []
        carl = Params.Params.carlength
        for i in range(4):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(random.uniform(0,11)+carl)
                elif msg[i] == 1:
                    res.append(random.uniform(11.1,27)+carl)
                else:
                    res.append(random.uniform(27.1,45)+carl)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(random.uniform(-0.1,-10))
                else:
                    res.append(random.uniform(0.1,10))

        for i in range(4,8):
            if i % 2 == 0:
                if msg[i] == 0:
                    res.append(random.uniform(-0, -11)-carl)
                elif msg[i] == 1:
                    res.append(random.uniform(-11.1, -27)-carl)
                else:
                    res.append(random.uniform(-27.1, -45)-carl)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(random.uniform(0.1,10))
                else:
                    res.append(random.uniform(-0.1,-10))

        for i in range(8,14):
            if i % 2 == 0:
                if msg[i] == 0:
                    res.append(random.uniform(0, 11)+carl)
                elif msg[i] == 1:
                    res.append(random.uniform(11.1, 27)+carl)
                else:
                    res.append(random.uniform(27.1, 45)+carl)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(random.uniform(-0.1, -10))
                else:
                    res.append(random.uniform(0.1, 10))

        for i in range(14,18):
            if i % 2 == 0:
                if msg[i] == 0:
                    res.append(random.uniform(-0, -11)-carl)
                elif msg[i] == 1:
                    res.append(random.uniform(-11.1, -27)-carl)
                else:
                    res.append(random.uniform(-27.1, -45)-carl)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(random.uniform(0.1, 10))
                else:
                    res.append(random.uniform(-0.1, -10))

        res.append(lane)

        return res

    def get_contMessage(self,msg,lane):
        res = []
        for i in range(4):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(11)
                elif msg[i] == 1:
                    res.append(27)
                else:
                    res.append(40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(-1)
                else:
                    res.append(1)

        for i in range(4,8):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(-11)
                elif msg[i] == 1:
                    res.append(-27)
                else:
                    res.append(-40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(1)
                else:
                    res.append(-1)

        for i in range(8,14):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(11)
                elif msg[i] == 1:
                    res.append(27)
                else:
                    res.append(40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(-1)
                else:
                    res.append(1)

        for i in range(14,18):
            if i%2 == 0:
                if msg[i] == 0:
                    res.append(-11)
                elif msg[i] == 1:
                    res.append(-27)
                else:
                    res.append(-40)
            else:
                if msg[i] == 0:
                    res.append(0)
                elif msg[i] == 1:
                    res.append(1)
                else:
                    res.append(-1)

        res.append(lane)

        return res

    def d_char(self, msg, i):
        if (msg.msg[i] == 'close'):
            return '0'
        elif (msg.msg[i] == 'nominal'):
            return '1'
        elif (msg.msg[i] == 'far'):
            return '2'
        else:
            print('cant find ' + msg.msg[i] + ' in d_char() options')
            return '*'

    def v_char(self, msg, i):
        if (msg.msg[i] == 'stable'):
            return '0'
        elif (msg.msg[i] == 'approaching'):
            return '1'
        elif (msg.msg[i] == 'retreating'):
            return '2'
        else:
            print('cant find ' + msg.msg[i] + ' in v_char() options')
            return '*'

    def lane_char(self, msg, i):
        if (msg.msg[i] == 'right'):
            return '0'
        elif (msg.msg[i] == 'center1'):
            return '1'
        elif (msg.msg[i] == 'center2'):
            return '2'
        elif (msg.msg[i] == 'center3'):
            return '3'
        elif (msg.msg[i] == 'left'):
            return '4'
        else:
            print('cant find ' + msg.msg[i] + ' in lane_char() options')
            return '*'

    def classify_dist(self, distance):
        d = abs(distance)
        if d <= Params.Params.close_distance:
            return 'close'
        elif d <= Params.Params.far_distance:
            return 'nominal'
        else:
            return 'far'
        return 'error in classify_dist()'

    def classify_vel(self, d, v):
        if abs(v) < 0.1:
            return 'stable'
        elif (d >= 0 and v <= -0.1) or (d < 0 and v >= 0.1):
            return 'approaching'
        elif (d >= 0 and v >= 0.1) or (d < 0 and v <= -0.1):
            return 'retreating'
        return 'error in classify_vel()'

    def classify_lane(self, lane):
        if lane == 4:
            return 'left'
        elif lane == 3:
            return 'center3'
        elif lane == 2:
            return 'center2'
        elif lane == 1:
            return 'center1'
        elif lane == 0:
            return 'right'
        return 'error in classify_lane()'

    def printState(self, file):
        str = '' + self.get_reward() + '\t'
        for temp in self.cars:
            str += temp.position_x + '\t' + temp.position_y + '\t' + temp.velocity_x + '\t' + temp.current_action + '\t' + temp.Mode + '\t' + self.get_Messgae(
                temp) + '\t'
        file.write(str)

    def get_reward(self):
        car0 = self.cars[0]
        wc = 1000
        wv = 1 #AAMAS 2
        we = 10 #AAMAS 10
        wh = 4 #AAMAS 7.5
        c = 0
        if self.collision_check_new():
            c = -1
        v = (car0.velocity_x - Params.Params.nominal_speed) * 3.6 / 88
        if car0.velocity_x > 90:
            v = (40 - car0.velocity_x) * 3.6 / 10

        e = 0
        if car0.effort == 0:
            e = 0
        elif car0.effort == 1:
            e = -0.25
        elif car0.effort == 2:
            e = -0.5
        elif car0.effort == 3:
            e = -1
        elif car0.effort == 4:
            e = -1
        else:
            print('error assigning e in get_reward()')

        h = 0
        aaa = Message.Message(self.get_Message(car0))
        if aaa.fc_d < 11:
            h = -0.5
        elif aaa.fc_d <= 27:
            h = 0
        elif aaa.fc_d > 27:
            h = 0.5
        else:
            print('error assigning h in get_reward()')

        return wc * c + wv * v + we * e + wh * h

    def str2int(self, x, base):
        temp = 0
        ind = len(x) - 1;
        for i in range(0, len(x)):
            temp += pow(base, ind) * int(x[i])
            ind -= 1
        return temp