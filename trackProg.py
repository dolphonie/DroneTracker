#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Begin autonomous flight when quad is armed and enters guided mode
"""

from dronekit import connect, VehicleMode, LocationLocal, LocationGlobal
from pymavlink import mavutil
from myVehicle import MyVehicle
from utils import *
import time, math, lrs
import numpy as np

####################################################################
# Constants

SLEEP_TIME = 2
ALT = 1
GPS_PORT = "com7"

MAVLINK_ADDR = ":14551"                 # UDP server
#MAVLINK_ADDR = "/dev/ttyUSB0,57600"     # Ubuntu USB serial adapter
#MAVLINK_ADDR = "/dev/ttyUSB0,1500000"     # Ubuntu USB serial adapter

SIDE = 10
THETA = math.pi/2 + math.pi/8
SIN = SIDE*math.sin(THETA)
COS = SIDE*math.cos(THETA)
CIRCLE_TIME = 30 #seconds
CIRCLE_DIAMETER = 20

SQUARE_COURSE = [LocationLocal(SIN,COS,-ALT),
                 LocationLocal(SIN-COS,COS+SIN,-ALT),
                 LocationLocal(-COS,SIN,-ALT),
                 LocationLocal(0, 0, -ALT)]
####################################################################
# Classes
class AbortAuton(Exception):
    def __str__(self):
        return "Abort Autonomous Flight"


####################################################################
# Subroutines
def isGuided(vehicle):
    return vehicle.mode.name=="GUIDED"

def isArmedGuided(vehicle):
    return vehicle.armed and isGuided(vehicle)

def sleep(vehicle, sleepTime=SLEEP_TIME):
    if not isArmedGuided(vehicle):
        print("No longer armed and GUIDED: Aborting Mission")
        raise AbortAuton()
    else:
        time.sleep(sleepTime)

def waitUntilArmedGuided(vehicle):
    print("Waiting for Quad to be armed in GUIDED mode")
    while not (isGuided(vehicle) and vehicle.armed):
        time.sleep(SLEEP_TIME)
    print("Quad armed and in GUIDED mode")

def waitForRestartMission(vehicle):
    print("Switch out of guided mode to allow restarting mission")
    while isGuided(vehicle):
        time.sleep(SLEEP_TIME)

def waitForAutopilotVersion(vehicle):
    # I'm not exactly sure what this does
    print("Waiting for autopilot version")
    vehicle.wait_ready("autopilot_version")

def connectToQuad():
    print("connect to quad")
    vehicle = connect(MAVLINK_ADDR, wait_ready=True, vehicle_class=MyVehicle)

    waitForAutopilotVersion(vehicle)

    return vehicle

def takeOff(vehicle, targetAlt):
    print("take off and climb to: %s" % targetAlt)
    vehicle.simple_takeoff(targetAlt)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print("current altitude: %s" % alt) 
        if alt>=targetAlt*.95:
            print("take off reached target altitude: %s" % alt)
            break
        sleep(vehicle)

def distanceHorizontal(locationLocal1, locationLocal2):
    distance = math.sqrt(
            (locationLocal1.north - locationLocal2.north)**2 +
            (locationLocal1.east - locationLocal2.east)**2
        )
    return distance

def gotoLocationLocal(vehicle, locLocal):
    print("goto %s" % locLocal)
    vehicle.gotoLocationLocal(locLocal)

    # wait until we arrive at destination
    while True:
        curLoc = vehicle.getLocationLocal()
        distance = distanceHorizontal(curLoc, locLocal)
        print("Distance to target location: %s" % distance)
        if (distance<1):
            print("Arrived at target location")
            break
        sleep(vehicle)

def followCourse(vehicle, course):
    print("Begin mission")

    takeOff(vehicle, ALT)

    for p in course:
        gotoLocationLocal(vehicle, p)

    print("End mission")


def followMe(vehicle, gps):
    takeOff(vehicle, ALT)

    while True:
        targetGlob = LocationGlobal(gps.latitude, gps.longitude,
                                    gps.altitude)
        print("global target: {},{}".format(gps.latitude,gps.longitude))
        targetLocal = globalMinusGlobalToLocal(targetGlob, vehicle.myHome)
        targetLocal.down = -ALT
        gotoLocationLocal(vehicle, targetLocal)
        sleep(vehicle, sleepTime=1)

#ex code
def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """
    Move vehicle in direction based on specified velocity vectors.
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111000111, # type_mask (only speeds enabled)
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)


    # send command to vehicle on 1 Hz cycle
    
    vehicle.send_mavlink(msg)
    time.sleep(1)

#ex code
def condition_yaw(heading, relative=False):
    if relative:
        is_relative=1 #yaw relative to direction of travel
    else:
        is_relative=0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def followCircleVelocity(vehicle):
    print('Welcome to Patrick Air! Please fasten your barf bags securely to your mouth')

    takeOff(vehicle, ALT)

    gotoLocationLocal(vehicle,LocationLocal(0,CIRCLE_DIAMETER/2,-ALT))
    #while math.fabs(vehicle.getLocationLocal().east-CIRCLE_RADIUS)>=3:
    #    sleep(vehicle,sleepTime=.2)

    #condition_yaw(0);
    #while math.fabs(vehicle.getLocationLocal().east-CIRCLE_RADIUS)>=math.pi/3:

    print('Vehicle at edge of circle')
    while True:
        circleEast,circleNorth = circlingObjectPosition(time.clock())
        vEast = .1*(circleEast-vehicle.getLocationLocal().east)
        vNorth = .1*(circleNorth -vehicle.getLocationLocal().north)
        send_ned_velocity(vNorth,vEast, 0);
        
        #target = LocationLocal(circleNorth,circleEast, -ALT)
        #gotoLocationLocal(vehicle,target)
        sleep(vehicle, sleepTime =.05)

def trackQuad(vehicle):
    print('Welcome to Patrick Air! Please fasten your barf bags securely to your mouth')
    takeOff(vehicle, ALT)
    while True:
        frame = lrs.getFrame()        
        vEast =  vehicle.getLocationLocal().east+frame[3] #xpos
        vNorth = vehicle.getLocationLocal().north + frame[5] #zpos
        vDown =vehicle.getLocationLocal().down + frame[4] #ypos
        send_ned_velocity(vNorth,vEast, vDown);
        
        #target = LocationLocal(circleNorth,circleEast, -ALT)
        #gotoLocationLocal(vehicle,target)
        sleep(vehicle, sleepTime =.05)

def followMeVelocity(vehicle,gps):
    print('Welcome to Patrick Air! Please fasten your barf bags securely to your mouth')
    takeOff(vehicle, ALT)


#Starts at 0rad at t=0 and circles
def circlingObjectPosition(time):
    angleTraveled = time * 2* math.pi/CIRCLE_TIME
    return CIRCLE_DIAMETER*math.cos(angleTraveled),CIRCLE_DIAMETER*math.sin(angleTraveled)    
    


####################################################################
# begin execution
vehicle = connectToQuad()

mode = "track quad"

# Start GPS right away to give it time to initialize
# Also, discover initialization problems right away
if mode=="follow me":
    gps = Gps(GPS_PORT)

while True:
    waitUntilArmedGuided(vehicle)
    try:
        if mode=="follow course":
            followCourse(vehicle, SQUARE_COURSE)
        elif mode=="follow me":
            followMe(vehicle, gps)
        elif mode=="follow circle":
            followCircleVelocity(vehicle)
        elif mode=="track quad":
            trackQuad(vehicle)
        else:
            print("Unrecognized mode: {}".format(mode))
            break
    except AbortAuton:
        print("Mission aborted")
        
    if (mode=="follow me"): break;
    
    waitForRestartMission(vehicle)






