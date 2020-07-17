import random
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import util.mol_conv as mconv
import util.trainer as tr
from torch_geometric.data import DataLoader
from model.GCN import GCN


num_epochs = 10000
batch_size = 128
lr = 5e-4


def test(model, data_loader):
    model.eval()
    targets = list()
    preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            pred = model(batch)
            targets.append(batch.y.cpu().numpy())
            preds.append(pred.cpu().numpy())

    return numpy.vstack(targets), numpy.vstack(preds)


data_list = mconv.read_dataset('data/esol_weight.csv')
targets = [x.y.item() for x in data_list]
mean = numpy.mean(targets)
std = numpy.std(targets)
for x in data_list:
    x.y = (x.y - mean) / std

random.shuffle(data_list)
criterion = nn.L1Loss()
data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)


model = GCN(mconv.num_atm_feats, 1, readout='mean').cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses = numpy.empty((num_epochs, 1))
for epoch in range(0, num_epochs):
    train_losses[epoch, 0] = tr.train(model, optimizer, data_loader, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, num_epochs, train_losses[epoch, 0]))
targets, preds = test(model, data_loader)
numpy.savetxt('train_weight_mean.csv', train_losses, delimiter=',')
numpy.savetxt('reg_weight_mean.csv', numpy.hstack((targets, preds)), delimiter=',')


model = GCN(mconv.num_atm_feats, 1, readout='max').cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses = numpy.empty((num_epochs, 1))
for epoch in range(0, num_epochs):
    train_losses[epoch, 0] = tr.train(model, optimizer, data_loader, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, num_epochs, train_losses[epoch, 0]))
targets, preds = test(model, data_loader)
numpy.savetxt('train_weight_max.csv', train_losses, delimiter=',')
numpy.savetxt('reg_weight_max.csv', numpy.hstack((targets, preds)), delimiter=',')


model = GCN(mconv.num_atm_feats, 1, readout='add').cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
train_losses = numpy.empty((num_epochs, 1))
for epoch in range(0, num_epochs):
    train_losses[epoch, 0] = tr.train(model, optimizer, data_loader, criterion)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, num_epochs, train_losses[epoch, 0]))
targets, preds = test(model, data_loader)
numpy.savetxt('train_weight_add.csv', train_losses, delimiter=',')
numpy.savetxt('reg_weight_add.csv', numpy.hstack((targets, preds)), delimiter=',')
