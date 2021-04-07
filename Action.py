import Params


##Action class in order to implement different actions
# ax is the longitudenal acceleration value
# vy is the lateral velocity

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

#There are 7 different actions defined below
#-------------------------------------------
#-------------------------------------------
##Changing acceleration actions
# 1. Maintain, 2. Accelerate, 3. Decelerate, 4. Hard Accelerate, 5. Hard Decelerate
# For these, lateral speed is zero and longitudenal accileration values are defined in Params.py
#-------------------------------------------
##Changing lane actions
# 1. Move Left, 2. Move Right
# For these, longitudenal acceleration is zero
# Since it is assumed that lane changes are completed in 1s, lateral speed is equal to the lanewidth
#-------------------------------------------

maintain = Action("maintain", 0, 0)
accelerate = Action("accelerate", Params.Params.accel_rate, 0)
decelerate = Action("decelerate", Params.Params.decel_rate, 0);
hard_accelerate = Action("hard_accelerate", Params.Params.hard_accel_rate, 0);
hard_decelerate = Action("hard_decelerate", Params.Params.hard_decel_rate, 0);
moveleft = Action("move left", 0, Params.Params.lanewidth);
moveright = Action("move right", 0, -Params.Params.lanewidth);


