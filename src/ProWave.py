import pandas as pd
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

VOCAB_SIZE = 30
CLANS = 10
FAMILIES = 100
LENGTH = 512
LENGTH_OUT = 514

with open('./data/train_red2.pkl', 'rb') as f:
    train_data = pkl.load(f)
with open('./data/valid_red2.pkl', 'rb') as f:
    valid_data = pkl.load(f)
print('Loaded data')

from utils import get_data_input
X_train = get_data_input(train_data, LENGTH)
X_valid = get_data_input(valid_data, LENGTH)

from utils import get_data_output
Y_train = get_data_output(train_data, LENGTH_OUT)
Y_valid = get_data_output(valid_data, LENGTH_OUT)

from torch.utils.data import TensorDataset, DataLoader
EPOCHS = 30
BATCH_SIZE = 20

d_train = torch.tensor([pd.to_numeric(x[0]) for x in X_train.values]).to(torch.long)
d_targets = torch.tensor([pd.to_numeric(y[0]) for y in Y_train.values]).to(torch.long)
train_dataset = TensorDataset(d_train, d_targets)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 16)

v_train = torch.tensor([pd.to_numeric(x[0]) for x in X_valid.values]).to(torch.long)
v_targets = torch.tensor([pd.to_numeric(y[0]) for y in Y_valid.values]).to(torch.long)
valid_dataset = TensorDataset(v_train, v_targets)
valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 8)
print('Transformed data')

from model import ProWaveNet, nll_loss
netProWave = ProWaveNet( num_time_samples = (512 * 512) + 510 , num_layers = 16, num_hidden = 30, num_channels = 30, num_classes  = 30, num_blocks   = 4, kernel_size  = 2)
netProWave.criterion = nll_loss
netProWave.optimizer = optim.Adam(netProWave.parameters(), lr=0.001)
netProWave.scheduler = optim.lr_scheduler.StepLR(netProWave.optimizer, step_size = 5, gamma = 0.5)
print('Created model')

netProWave.train_WaveNet(train_loader, valid_loader, num_epochs = EPOCHS, disp_interval = 1, vocab_size = 30)

with open('./pretrained/netProWave_30Epochs', 'wb') as f:
    torch.save(netProWave.state_dict(), f)
