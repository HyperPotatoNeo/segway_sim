#!/usr/bin/env python3
import sys
import rospy
import numpy as np
from segway_sim.msg import stateAndAction
import rospkg
import csv

CALLBACK_COUNT = 0
filename = 'dagger1.csv'

with open(rospkg.RosPack().get_path('segway_sim') + '/src/pyFun/dataset/' + filename, "a") as fp:
	wr = csv.writer(fp, dialect='excel')
	
	def callback(state_and_action):
		state = state_and_action.cur_state.stateVec
		action = state_and_action.cur_input.inputVec
		data = state+action
		wr.writerow(data)

	rospy.init_node('data_collect', anonymous=True)
	rospy.Subscriber('/segway_sim/state_and_action', stateAndAction, callback)

	rospy.spin()