import numpy
import random
import torch.nn as nn
import util.mol_conv as mconv
import util.trainer as tr
from model.GIN import GIN


num_epochs = 300
batch_size = 128
num_classes = 2


data_list = mconv.read_dataset('data/bbbp.csv')
random.shuffle(data_list)
criterion = nn.CrossEntropyLoss()


avg_test_loss_gcn = numpy.empty(6)
avg_test_loss_gat = numpy.empty(6)


print('--------- mean ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='mean').cuda()
acc_mean, f1_mean = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('--------- max ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='max').cuda()
acc_max, f1_max = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('--------- attn ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='attn').cuda()
acc_attn, f1_attn = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('--------- lstm ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='lstm').cuda()
acc_lstm, f1_lstm = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('--------- add ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='sum').cuda()
acc_add, f1_add = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('--------- un_attn ---------')
model = GIN(mconv.num_atm_feats, num_classes, readout='uattn').cuda()
acc_un_attn, f1_un_attn = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, 5e-5, 1e-4, reg=False)

print('------------ GIN ------------')
print(acc_mean, f1_mean)
print(acc_max, f1_max)
print(acc_attn, f1_attn)
print(acc_lstm, f1_lstm)
print(acc_add, f1_add)
print(acc_un_attn, f1_un_attn)
