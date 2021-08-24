import Params, Action
import random

class Car:

    id = 0
    lane = 2
    position_x = 0
    prevprev = 0
    prevprevprev = 0
    prev_position_x = 0
    position_y = 0
    velocity_x = 0
    current_action = Action.maintain
    prev_action = Action.maintain
    effort = 0
    lanechange_flag = False
    already_change = False
    Mode = 0
    LC_times = 0

    def __init__(self, *args):

        if len(args) == 1:
            foundOneArg = True
            theOnlyArg = args[0]
        elif len(args) == 2:
            foundOneArg = False


        if foundOneArg:
            self.init2(theOnlyArg)
        else:
            self.init1(args[0], args[1])

    def init1(self, id, policy):
        self.distf = 0
        self.acc = 0
        self.id = id
        self.lane = 2
        self.prevprevprev = 0
        self.prevprev = 0
        self.position_x = 0
        self.prev_position_x = 0
        self.position_y = self.lane*Params.Params.lanewidth+0.5*Params.Params.lanewidth
        self.policy = policy
        speed_randomness = 5
        self.velocity_x = 7+speed_randomness*random.uniform(0.0,1.0)
        self.current_action = Action.maintain
        self.prev_action = Action.maintain
        self.effort = 0
        self.lanechange_flag = False
        self.already_change = False
        self.Mode = 0
        self.LC_times = 0

    def init2(self, copy):
        self.distf = 0
        self.acc = 0
        self.id = copy.id
        self.lane = copy.lane
        self.prevprevprev = copy.prevprevprec
        self.prevprev = copy.prevprev
        self.prev_position_x = copy.prev_position_x
        self.position_x = copy.position_x
        self.position_y = copy.position_y
        self.velocity_x = copy.velocity_x
        self.current_action = copy.current_action
        self.prev_action = copy.prev_action
        self.effort = copy.effort
        self.lanechange_flag = copy.lanechange_flag
        self.policy = copy.policy
        self.Mode = copy.Mode
        self.LC_times = copy.LC_times

    def equals(self, check):
        if self.id == check.id:
            return True
        return False

    def setAction(self, act, msg, msg_num, Switch):
        check = 0
        if self.lanechange_flag==False:
            self.already_change = False
            if (act.equals(Action.moveleft) and self.lane==4) or (act.equals(Action.moveright) and self.lane == 0):
                check = 1

            if Switch:
                cur = 0
                #currently free

            if check == 1:
                self.effort = 4
            elif act.equals(Action.maintain):
                self.effort = 0
            else:
                self.effort = 1
                if act.equals(Action.hard_decelerate) or act.equals(Action.hard_accelerate):
                    self.effort = 2
                if act.equals(Action.moveleft) or act.equals(Action.moveright):
                    self.lanechange_flag = True
                    self.effort = 3
                    self.LC_times += 1

            self.prev_action = self.current_action
            self.current_action = act

        return check

    def updateMotionNew(self, msg, car0, msg_num, Switch, action):

        if action == 0:
            act = Action.maintain
        elif action == 1:
            act = Action.accelerate
        elif action == 2:
            act = Action.decelerate
        elif action == 3:
            act = Action.hard_accelerate
        elif action == 4:
            act = Action.hard_decelerate
        elif action == 5:
            act = Action.moveleft
        else:
            act = Action.moveright

        check = self.setAction(act, msg, msg_num, Switch)
        epsilon = 0.1
        if check == 1:
            self.already_change = True
        prev = self.velocity_x
        if self.lanechange_flag == True:
            self.position_y += self.current_action.vy * Params.Params.timestep
            signvy = 0
            if self.current_action.vy > 0:
                signvy = 1
            elif self.current_action.vy < 0:
                signvy = -1

            if (signvy > 0 and (self.position_y - (self.lane + 1) * Params.Params.lanewidth) >= 0) or (
                    signvy < 0 and (self.position_y - (self.lane) * Params.Params.lanewidth) <= 0):
                if self.already_change == False:
                    self.lane += signvy
                    self.already_change = True

            if self.already_change and (abs(self.position_y - ((self.lane + 0.5) * Params.Params.lanewidth)) < epsilon):
                self.lanechange_flag = False

        elif self.current_action.equals(Action.accelerate) and self.velocity_x < (Params.Params.max_speed - epsilon):
            self.velocity_x += (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax) * Params.Params.timestep
            self.acc = (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax)

        elif self.current_action.equals(Action.decelerate) and self.velocity_x > (Params.Params.min_speed + epsilon):
            self.velocity_x += (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax) * Params.Params.timestep
            self.acc = (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax)

        elif self.current_action.equals(Action.hard_accelerate) and self.velocity_x < (
            Params.Params.max_speed - epsilon):
            self.velocity_x += (-abs(0.3 * random.gauss(0, 1)) + self.current_action.ax) * Params.Params.timestep
            self.acc = (-abs(0.3 * random.gauss(0, 1)) + self.current_action.ax)

        elif self.current_action.equals(Action.hard_decelerate) and self.velocity_x > (
            Params.Params.min_speed + epsilon):
            self.velocity_x += (abs(0.3 * random.gauss(0, 1)) + self.current_action.ax) * Params.Params.timestep
            self.acc = (abs(0.3 * random.gauss(0, 1)) + self.current_action.ax)

        elif self.current_action.equals(Action.maintain) and (
            self.velocity_x < (Params.Params.max_speed - epsilon)) and (
            self.velocity_x > (Params.Params.min_speed + epsilon)):
            self.acc = random.gauss(0.012, 0.138)
            self.velocity_x += self.acc*Params.Params.timestep

        if self.velocity_x < Params.Params.min_speed:
            self.velocity_x = Params.Params.min_speed
        elif self.velocity_x > Params.Params.max_speed:
            self.velocity_x = Params.Params.max_speed

        self.prevprevprev = self.prevprev
        self.prevprev = self.prev_position_x
        self.prev_position_x = self.position_x

        if prev+self.acc > Params.Params.min_speed:
            self.position_x += (prev+0.5*self.acc)*Params.Params.timestep
        elif prev == Params.Params.min_speed:
            self.position_x += (prev) * Params.Params.timestep
        else:
            self.position_x += (prev+0.5*(Params.Params.min_speed-prev))*Params.Params.timestep

        if self.position_x < -(Params.Params.init_size + 100):
            self.position_x += 2 * (Params.Params.init_size + 100)
        elif self.position_x > (Params.Params.init_size + 100):
            self.position_x += -2 * (Params.Params.init_size + 100)

    def updateMotionLevel(self, msg, car0, msg_num, Switch, action):

        if action == 0:
            act = Action.maintain
        elif action == 1:
            act = Action.accelerate
        elif action == 2:
            act = Action.decelerate
        elif action == 3:
            act = Action.hard_accelerate
        elif action == 4:
            act = Action.hard_decelerate
        elif action == 5:
            act = Action.moveleft
        else:
            act = Action.moveright

        check = self.setAction(act, msg, msg_num, Switch)
        epsilon = 0.1
        if check == 1:
            self.already_change = True
        prev = self.velocity_x
        if self.lanechange_flag == True:
            self.position_y += self.current_action.vy * Params.Params.timestep
            signvy = 0
            if self.current_action.vy > 0:
                signvy = 1
            elif self.current_action.vy < 0:
                signvy = -1

            if (signvy > 0 and (self.position_y - (self.lane + 1) * Params.Params.lanewidth) >= 0) or (
                    signvy < 0 and (self.position_y - (self.lane) * Params.Params.lanewidth) <= 0):
                if self.already_change == False:
                    self.lane += signvy
                    self.already_change = True

            if self.already_change and (abs(self.position_y - ((self.lane + 0.5) * Params.Params.lanewidth)) < epsilon):
                self.lanechange_flag = False

        elif self.current_action.equals(Action.accelerate) and self.velocity_x < (Params.Params.max_speed - epsilon):
            self.velocity_x += (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax) * Params.Params.timestep
            self.acc = (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax)
        elif self.current_action.equals(Action.decelerate) and self.velocity_x > (Params.Params.min_speed + epsilon):
            self.velocity_x += (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax) * Params.Params.timestep
            self.acc = (2 * random.uniform(0.0, 1.0) - 1 + self.current_action.ax)
        elif self.current_action.equals(Action.hard_accelerate) and self.velocity_x < (
            Params.Params.max_speed - epsilon):
            self.velocity_x += (-abs(0.3 * random.gauss(0, 1)) + self.current_action.ax) * Params.Params.timestep
            self.acc = (-abs(0.3 * random.gauss(0, 1)) + self.current_action.ax)
        elif self.current_action.equals(Action.hard_decelerate) and self.velocity_x > (
            Params.Params.min_speed + epsilon):
            self.velocity_x += (abs(0.3 * random.gauss(0, 1)) + self.current_action.ax) * Params.Params.timestep
            self.acc = (abs(0.3 * random.gauss(0, 1)) + self.current_action.ax)
        elif self.current_action.equals(Action.maintain) and (
            self.velocity_x < (Params.Params.max_speed - epsilon)) and (
            self.velocity_x > (Params.Params.min_speed + epsilon)):
            self.velocity_x += random.gauss(0.012, 0.138) * Params.Params.timestep
            self.acc = random.gauss(0.012, 0.138)

        if self.velocity_x < Params.Params.min_speed:
            self.velocity_x = Params.Params.min_speed
        elif self.velocity_x > Params.Params.max_speed:
            self.velocity_x = Params.Params.max_speed

        self.prevprevprev = self.prevprev
        self.prevprev = self.prev_position_x
        self.prev_position_x = self.position_x
        self.position_x += (prev+0.5*self.acc)*Params.Params.timestep
        if self.position_x < -(Params.Params.init_size + 100):
            self.position_x += 2 * (Params.Params.init_size + 100)
        elif self.position_x > (Params.Params.init_size + 100):
            self.position_x += -2 * (Params.Params.init_size + 100)


    def updateMotion(self, msg, car0, msg_num, Switch):
        if msg.fc_v < -0.05 and msg.fc_d < 11:
            action = Action.hard_decelerate
        elif (msg.fc_v < -0.05 and msg.fc_d <= 27 and msg.fc_d >= 11) or (
                msg.fc_v > -0.05 and msg.fc_v < 0.05 and msg.fc_d < 11):
            action = Action.decelerate
        elif (msg.fc_v > 0.05 and msg.fc_d <= 27 and msg.fc_d >= 11) or (msg.fc_d > 27):
            action = Action.accelerate
        else:
            action = Action.maintain

        check = self.setAction(action, msg, msg_num, Switch)
        epsilon = 0.1
        if check == 1:
            self.already_change = True
        prev = self.velocity_x
        if self.lanechange_flag == True:
            self.position_y += self.current_action.vy*Params.Params.timestep
            signvy = 0
            if self.current_action.vy > 0:
                signvy = 1
            elif self.current_action.vy < 0:
                signvy = -1

            if (signvy>0 and (self.position_y-(self.lane+1)*Params.Params.lanewidth)>=0) or (signvy<0 and (self.position_y-(self.lane)*Params.Params.lanewidth)<=0):
                if self.already_change == False:
                    self.lane += signvy
                    self.already_change = True

            if self.already_change and (abs(self.position_y-((self.lane+0.5)*Params.Params.lanewidth)) < epsilon):
                self.lanechange_flag = False

        elif self.current_action.equals(Action.accelerate) and self.velocity_x<(Params.Params.max_speed-epsilon):
            self.velocity_x += (2*random.uniform(0.0,1.0)-1+self.current_action.ax)*Params.Params.timestep
            self.acc = (2*random.uniform(0.0,1.0)-1+self.current_action.ax)
        elif self.current_action.equals(Action.decelerate) and self.velocity_x>(Params.Params.min_speed+epsilon):
            self.velocity_x += (2*random.uniform(0.0,1.0)-1+self.current_action.ax)*Params.Params.timestep
            self.acc = (2*random.uniform(0.0,1.0)-1+self.current_action.ax)
        elif self.current_action.equals(Action.hard_accelerate) and self.velocity_x<(Params.Params.max_speed-epsilon):
            self.velocity_x += (-abs(0.3*random.gauss(0,1))+self.current_action.ax)*Params.Params.timestep
            self.acc = (-abs(0.3*random.gauss(0,1))+self.current_action.ax)
        elif self.current_action.equals(Action.hard_decelerate) and self.velocity_x>(Params.Params.min_speed+epsilon):
            self.velocity_x += (abs(0.3*random.gauss(0,1))+self.current_action.ax)*Params.Params.timestep
            self.acc = (abs(0.3*random.gauss(0,1))+self.current_action.ax)
        elif self.current_action.equals(Action.maintain) and (self.velocity_x<(Params.Params.max_speed-epsilon)) and (self.velocity_x>(Params.Params.min_speed+epsilon)):
            self.velocity_x += random.gauss(0.012, 0.138)*Params.Params.timestep
            self.acc = random.gauss(0.012, 0.138)
        if self.velocity_x < Params.Params.min_speed:
            self.velocity_x = Params.Params.min_speed
        elif self.velocity_x > 11:
            self.velocity_x = 11

        self.prev_position_x = self.position_x

        if prev > Params.Params.min_speed:
            self.position_x += (prev + 0.5 * self.acc) * Params.Params.timestep
        else:
            self.position_x += (prev) * Params.Params.timestep

        if self.position_x < -(Params.Params.init_size+100):
            self.position_x += 2*(Params.Params.init_size+100)
        elif self.position_x > (Params.Params.init_size+100):
            self.position_x += -2*(Params.Params.init_size+100)

