import numpy as np
import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.astype(np.float32))
        self.y = torch.tensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    



class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, num_stacked_layers, device):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lstm = nn.LSTM(hidden_size1, hidden_size2, num_stacked_layers, batch_first=True)
        self.fc2 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.num_stacked_layers = num_stacked_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size1).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size1).to(self.device)

        out = self.fc1(x)
        out = out.transpose(1, -1) 
        # out shape = [1,128]
        out, _ = self.lstm(out) 
        # out.shape = [1,64]
        out = self.fc2(out)
        out = self.softmax(out)
        return out


def train_one_epoch(model, train_loader, loss_function, optimiser, device):
    model.train(True)
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        running_loss += loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return running_loss / len(train_loader)


def encode_labels(y_train):
    y_train_coded = []
    d = {'comedy':0, 'documentary':1, 'drama':2, 'other':3, 'short':4}
    for i in y_train:
        y_train_coded.append(d[i])
    return y_train_coded