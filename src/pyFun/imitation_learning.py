import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegwayDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data_list = pd.read_csv(file).values
        self.length = self.data_list.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X = self.data_list[idx,2:7]
        Y = self.data_list[idx,7:]
        return X, Y


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

def main(
        epochs=2048*4,
        batch_size=512,
        filename='dagger0.csv',
        load_model=None,
        checkpt_name='dagger0'
        ):
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 6}
    dataset = SegwayDataset(file='dataset/'+filename)
    training_generator = torch.utils.data.DataLoader(dataset, **params)
    
    model = MLP(5, 512).to(device).double()
    opt = optim.Adam(model.parameters(), 0.1)
    if(load_model is not None):
        checkpt = torch.load('weights/'+load_model)
        model.load_state_dict(checkpt['model'])
        opt.load_state_dict(checkpt['opt'])
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
            epoch_loss = 0
            if(epoch%2048==0):
                for g in opt.param_groups:
                    g['lr'] = g['lr']/5
            for X,Y in training_generator:
                opt.zero_grad()
                X = X.to(device).double()
                Y = Y.to(device).double()
                y_pred = model(X)
                loss = criterion(y_pred, Y)
                epoch_loss += loss.item()
                loss.backward()
                opt.step()
            print('EPOCH ',epoch,' LOSS: ',epoch_loss)

    torch.save({'model' : model.state_dict(), 'opt' : opt.state_dict()}, 'weights/'+checkpt_name)


if __name__=='__main__':
    main()