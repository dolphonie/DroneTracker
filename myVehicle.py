#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from dronekit import Vehicle
from utils import *

class MyVehicle(Vehicle):
    def __init__(self, *args):
        super(MyVehicle, self).__init__(*args)
        
        self.myHome = None

        self.add_attribute_listener("armed", self._armedChanged)

    def getValidLocation(self):
        loc = self.location.global_frame
        while not validLocation(loc):
            print("discarding invalid vehicle GPS location")
            time.sleep(.5)
            loc = self.location.global_frame
        return loc

    # whenever vehicle becomes armed, update home location
    def _armedChanged(self, vehicle, attr_name, value):
        if value==True:
            self.myHome = self.getValidLocation()

    def gotoLocationLocal(self, locLocal):
        self.simple_goto(globalPlusLocalToGlobal(self.myHome, locLocal))

    def getLocationLocal(self):
        return globalMinusGlobalToLocal(self.location.global_frame, self.myHome)



