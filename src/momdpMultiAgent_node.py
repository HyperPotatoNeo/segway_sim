#!/usr/bin/env python

import os
import datetime
import rospy
import numpy as np
import scipy
import pdb
import sys
import time
import matplotlib.pyplot as plt
import pickle
import sys
from uav_sim_ros.msg import state as stateDrone

sys.path.append(sys.path[0]+'/pyFun')
print("sys.path: ", sys.path)

from segway_sim.msg import goalSetAndState
import scipy.io as sio
import numpy as np
from MOMDP import MOMDP_TOQ, MOMDP_TOQ_d

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy
import math
homedir = os.path.expanduser("~")    

def main():
    # Initializa ROS node
    rospy.init_node("momdpSegway")

    # Initialize node parameters
    loop_rate      = 100.0
    dt             = 1.0/loop_rate
    rate           = rospy.Rate(loop_rate)
    x_start        = rospy.get_param("mpc/x_start")
    y_start        = rospy.get_param("mpc/y_start")
    expFlag        = rospy.get_param("momdpMultiAgent_node/expFlag")

    if expFlag == 1:
        from ambercortex_ros.msg import state
    else:
        from segway_sim.msg import state

    # Initialize subscriber and publisher
    stateMeasurements = StateMeasurements(x_start, y_start, expFlag, state) # this object read the true state (not the estimated)
    droneStateMeasurements = DroneStateMeasurements() # this object read the true state (not the estimated)
    
    publisher = rospy.Publisher('/segway_sim/visualization_marker_array', MarkerArray)
    markerArray = MarkerArray()
    goalSetAndState_pub = rospy.Publisher('goalSetAndState', goalSetAndState, queue_size=1)
    goalSetAndStateMsg  = goalSetAndState()   
    goalDroneSetAndState_pub = rospy.Publisher('droneGoalSetAndState', goalSetAndState, queue_size=1)
    goalDroneSetAndStateMsg  = goalSetAndState()   

    # Import MOMDPSegway
    pickle_in = open(sys.path[0]+'/pyFun/multiAgent/segway_7x7ug_2.pkl',"rb")
    momdpSegway     = pickle.load(pickle_in)
    momdpSegway.printLevel = 0
    pickle_in  = open(sys.path[0]+'/pyFun/multiAgent/drone_7x7ug_d_2.pkl',"rb")
    momdpDrone = pickle.load(pickle_in)
    momdpDrone.printLevel = 0

    # Init Environment
    loc        = (0, 0, 0, 1)
    initBelief = [0.3, 0.9, 0.85, 0.05]

    xt         = [0]
    ts          = 0
    verbose    = 0
    at = []

    momdpSegway.initZ(loc)
    momdpDrone.initZ(loc)
    bt = [momdpSegway.initBelief(initBelief)]
    markerArray = initMarkerArray(markerArray, momdpSegway)
    pSuccess = np.max(np.dot(momdpSegway.J[0, xt[0]].T, bt[-1])) # This is the probability of success
    print("============================ probability of success is: ", pSuccess)


    # Initialize MOMDPSegway
    momdpSegway.printLevel = 0
    segwayAgent = Agent(momdpSegway, goalSetAndState_pub, xt, bt)
    droneAgent  = Agent(momdpDrone, goalDroneSetAndState_pub, xt, bt)

    # Initialize parameters for while loop
    initFlag = 0
    decisionMaker = True
    deploySegway  = False
    deployDrone   = False
    print('=================== Readdy to start')
    while (not rospy.is_shutdown()):
        if stateMeasurements.state != []:
            # High-level decision maker
            if (decisionMaker == True):
                pSuccess = np.max(np.dot(momdpSegway.J[0, xt[0]].T, bt[-1])) # This is the probability of success
                if (pSuccess == 1):
                    deploySegway  = False
                    deployDrone   = False
                elif (pSuccess > 0.8):
                    deploySegway = True
                    segwayAgent.Update()
                    markerArray = updateObstacles(markerArray, segwayAgent.momdp, segwayAgent.bt[-2])
                else:
                    deployDrone = True
                    droneAgent.Update()
                    markerArray = updateObstacles(markerArray, segwayAgent.momdp, droneAgent.bt[-2])

                decisionMaker = False
                publisher.publish(markerArray)
            # read measurement
            xCurr = stateMeasurements.state
            xDroneCurr = droneStateMeasurements.state
            # Check if goal reached or reached the goal cell
            if (deploySegway==True) and segwayAgent.checkGoalReached(xCurr):
                segwayAgent.Update()
                # Update \alpha for obstacles: set \alpha = 1-prob beeing free
                markerArray = updateObstacles(markerArray, segwayAgent.momdp, segwayAgent.bt[-2])
                publisher.publish(markerArray)
                if (segwayAgent.xt[-1] == momdpSegway.goal[0]) or ( segwayAgent.t + 3.0 >= momdpSegway.V.shape[0] ):
                    deploySegway = False
                    decisionMaker = True
                    bt.append(segwayAgent.bt[-1])

            elif (deployDrone==True) and droneAgent.checkGoalReached(xDroneCurr):
                droneAgent.Update(forecast=False)
                # Update \alpha for obstacles: set \alpha = 1-prob beeing free
                markerArray = updateObstacles(markerArray, segwayAgent.momdp, droneAgent.bt[-2])
                publisher.publish(markerArray)
                if ( droneAgent.t + 3.0 >= momdpDrone.V.shape[0] ):
                    deployDrone = False
                    decisionMaker = True
                    bt.append(droneAgent.bt[-1])

            rate.sleep()
            
# ===============================================================================================================================
# ==================================================== END OF MAIN ==============================================================
# ===============================================================================================================================
class Agent():
    def __init__(self, momdp, pub, x0, b0):
        self.t = 0
        self.momdp = momdp
        self.goalSetAndStateMsg = goalSetAndState()
        self.pub = pub
        self.xt  = x0
        self.bt  = b0
        self.at  = []
    def checkGoalReached(self, x):
        return (x[0] >= self.boxNext[0]) and (x[0] <= self.boxNext[1]) and (x[1] >= self.boxNext[2]) and (x[1] <= self.boxNext[3])

    def Update(self, forecast=True):
        [action, coordXY, boxConstraints, self.boxNext] = self.momdp.updateMOMDP(self.t, self.xt[-1], self.bt[-1])
        self.at.append(action)
        self.xt.append(self.momdp.propagate(self.xt[-1], self.momdp.zt, self.at[-1]))
        self.t += 1
        self.goalSetAndStateMsg = pubHighLevel(self.goalSetAndStateMsg, coordXY, boxConstraints, self.boxNext, self.t, forecast)
        self.pub.publish(self.goalSetAndStateMsg) 

        # Update belief
        oMeas = self.momdp.getObservation(self.xt[-1], 3)
        self.bt.append( self.momdp.updateBelief(self.xt[-2], self.xt[-1], self.at[-1], oMeas, self.bt[-1]) )


def initMarkerArray(markerArray, momdpSegway):
    colorMarker = [1.0, 1.0, 0.0]
    [markerArray.markers.append(initMarker(momdpSegway, 0, i, colorMarker)) for i in range(0, momdpSegway.numUnKnownObst)]
    # Initialize marker array for goals
    colorMarker = [0.0, 1.0, 0.0]
    if momdpSegway.unGoal == False:
        markerArray.markers.append(initMarker(momdpSegway, 2, 0, colorMarker))
    else:
        [markerArray.markers.append(initMarker(momdpSegway, 2, i, colorMarker)) for i in range(0, momdpSegway.numUGoal)]
    
    colorMarker = [1.0, 1.0, 0.0]
    [markerArray.markers.append(initMarker(momdpSegway, 1, i, colorMarker)) for i in range(0, momdpSegway.numKnownObst)]
    # Set idx for marker array
    idx = 0
    for m in markerArray.markers:
        m.id = idx
        idx += 1
    return markerArray

def updateObstacles(markerArray, momdpSegway, bt):
    obstBelief = []
    [obstBelief.append( momdpSegway.computeBelief(bt, i) ) for i in range(0, momdpSegway.numUnKnownObst)]
    counter = 0
    for obsPr in obstBelief:
        markerArray.markers[counter].color.a = 1-obsPr
        counter += 1
    if momdpSegway.unGoal == True:
        goalBelief = []
        [goalBelief.append( momdpSegway.computeBelief(bt, i+momdpSegway.numObs) ) for i in range(0, momdpSegway.numUGoal)]
        for obsPr in goalBelief:
            markerArray.markers[counter].color.a = 1-obsPr
            counter += 1

    return markerArray

def initMarker(momdpSegway, knowObst, idx, colorMarker):
    ## Visualization
    if knowObst<2:
        cubeLength = 0.2;
        cubeHeigth = 0.2;
    else:
        cubeLength = 1.0;
        cubeHeigth = 0.00000001;

    marker = Marker()
    marker.header.frame_id = "/world"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = cubeLength
    marker.scale.y = cubeLength
    marker.scale.z = cubeHeigth
    marker.color.a = 1.0
    marker.color.r = colorMarker[0]
    marker.color.g = colorMarker[1]
    marker.color.b = colorMarker[2]
    marker.pose.orientation.w = 1.0

    if knowObst == 0:
        obstCoordinate = (momdpSegway.row_obs, momdpSegway.col_obs)
    elif knowObst == 1:
        obstCoordinate = np.where( (momdpSegway.gridVar < 0) )
    elif knowObst == 2:
        obstCoordinate = np.where( (momdpSegway.gridVar == 1) )

    marker.pose.position.x = obstCoordinate[1][idx] + momdpSegway.cellWidth/2
    marker.pose.position.y = np.shape(momdpSegway.gridVar)[0] - obstCoordinate[0][idx] - momdpSegway.cellHight/2.0
    marker.pose.position.z = 0
    return marker

def pubHighLevel(goalSetAndStateMsg, coordXY, boxConstraints, boxNext, t, forecast = True):
    if forecast == True and len(coordXY) > 2:
        goalSetAndStateMsg.x = (coordXY[1][0] + coordXY[2][0])/2
        goalSetAndStateMsg.y = (coordXY[1][1] + coordXY[2][1])/2
    else:
        goalSetAndStateMsg.x = coordXY[1][0]
        goalSetAndStateMsg.y = coordXY[1][1]
    
    goalSetAndStateMsg.xmin = boxConstraints[0]
    goalSetAndStateMsg.xmax = boxConstraints[1]
    goalSetAndStateMsg.ymin = boxConstraints[2]
    goalSetAndStateMsg.ymax = boxConstraints[3]   

    goalSetAndStateMsg.term_xmin = boxNext[0]
    goalSetAndStateMsg.term_xmax = boxNext[1]
    goalSetAndStateMsg.term_ymin = boxNext[2]
    goalSetAndStateMsg.term_ymax = boxNext[3]   

    goalSetAndStateMsg.highLevTime = t   

    return goalSetAndStateMsg


class StateMeasurements():
    def __init__(self, x_start, y_start, expFlag, state):
        self.state = []
        self.x_start = x_start
        self.y_start = y_start
        if expFlag == 1:
            rospy.Subscriber("state", state, self.state_callback_exp)
        else:
            rospy.Subscriber("state_true", state, self.state_callback_sim)

    def state_callback_exp(self, msg):
        self.state = [msg.state[0]+self.x_start, msg.state[1]+self.y_start, msg.state[2], msg.state[3], msg.state[4], msg.state[5], msg.state[6]]

    def state_callback_sim(self, msg):
        self.state = [msg.x, msg.y, msg.theta, msg.v, msg.thetaDot, msg.psi, msg.psiDot]

class DroneStateMeasurements():
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        rospy.Subscriber("/uav_sim_ros/state", stateDrone, self.droneState_callback)

    def droneState_callback(self, msg):
        self.state = np.array([msg.x, msg.y, msg.vbx, msg.vby])

if __name__ == "__main__":

    try:    
        main()
        
    except rospy.ROSInterruptException:
        pass
