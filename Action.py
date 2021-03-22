import Params

class Action:
    def __init__(self, action = '', ax = 0, vy = 0):
        self.action = action
        self.ax = ax
        self.vy = vy

    def equals(self, check):
        if self.ax == check.ax and self.vy == check.vy:
            return True
        else:
            return False

maintain = Action("maintain", 0, 0)
accelerate = Action("accelerate", Params.Params.accel_rate, 0)
decelerate = Action("decelerate", Params.Params.decel_rate, 0);
hard_accelerate = Action("hard_accelerate", Params.Params.hard_accel_rate, 0);
hard_decelerate = Action("hard_decelerate", Params.Params.hard_decel_rate, 0);
moveleft = Action("move left", 0, Params.Params.lanewidth);
moveright = Action("move right", 0, -Params.Params.lanewidth);


