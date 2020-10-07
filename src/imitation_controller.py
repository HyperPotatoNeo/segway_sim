#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from segway_sim.msg import stateAndAction
from segway_sim.msg import input as mpc_input
import rospkg
import rospy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def callback(state_and_action):
	state = state_and_action.cur_state.stateVec
	action = state_and_action.cur_input.inputVec
	
	x = torch.tensor(state[2:]).unsqueeze(0).to(device).double()
	pred_action = model(x)
	pred_action = pred_action.squeeze(0).cpu().tolist()
	print(pred_action, action)
	control = mpc_input()
	control.inputVec[0] = pred_action[0]
	control.inputVec[1] = pred_action[1]
	
	pub.publish(control)


checkpt_name = 'dagger0'
model = MLP(5, 512).to(device).double()
checkpt = torch.load(rospkg.RosPack().get_path('segway_sim')+'/src/pyFun/weights/'+checkpt_name)
model.load_state_dict(checkpt['model'])

rospy.init_node('imitation_controller', anonymous=True)
pub = rospy.Publisher('/segway_sim/mpc_input', mpc_input, queue_size=10)
rospy.Subscriber('/segway_sim/state_and_action', stateAndAction, callback)
rospy.spin()