from util.data import load_data
import numpy
import random
import torch.nn as nn
import util.trainer as tr
from model.GAT import GAT


num_epochs = 300
batch_size = 128
num_classes = 6
k = 5
lr = 5e-4
l2 = 1e-4


data_list, num_node_feats = load_data('COIL')
random.shuffle(data_list)
criterion = nn.CrossEntropyLoss()


avg_test_loss_gcn = numpy.empty(6)
avg_test_loss_gat = numpy.empty(6)


print('--------- mean ---------')
model = GAT(num_node_feats, num_classes, readout='mean').cuda()
acc_mean, f1_mean = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (mean): {:.4f}'.format(acc_mean))

print('--------- max ---------')
model = GAT(num_node_feats, num_classes, readout='max').cuda()
acc_max, f1_max = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (max): {:.4f}'.format(acc_max))

print('--------- attn ---------')
model = GAT(num_node_feats, num_classes, readout='attn').cuda()
acc_attn, f1_attn = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (attn): {:.4f}'.format(acc_attn))

print('--------- lstm ---------')
model = GAT(num_node_feats, num_classes, readout='lstm').cuda()
acc_lstm, f1_lstm = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (lstm): {:.4f}'.format(acc_lstm))

print('--------- add ---------')
model = GAT(num_node_feats, num_classes, readout='sum').cuda()
acc_add, f1_add = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (add): {:.4f}'.format(acc_add))

print('--------- un_attn ---------')
model = GAT(num_node_feats, num_classes, readout='uattn').cuda()
acc_un_attn, f1_un_attn = tr.cross_validation(data_list, k, model, criterion, num_epochs, batch_size, lr, l2, reg=False)
print('Test acc (un_attn): {:.4f}'.format(acc_un_attn))

print('------------ GAT ------------')
print(acc_mean, f1_mean)
print(acc_max, f1_max)
print(acc_attn, f1_attn)
print(acc_lstm, f1_lstm)
print(acc_add, f1_add)
print(acc_un_attn, f1_un_attn)
