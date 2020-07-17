import random
import numpy
import torch.nn as nn
import util.mol_conv as mconv
import util.trainer as tr
from model.GCN import GCN


num_epochs = 300
batch_size = 128
lr = 5e-5


data_list = mconv.read_dataset('data/qm9_hlg.csv')
targets = [x.y.item() for x in data_list]
mean = numpy.mean(targets)
std = numpy.std(targets)
for x in data_list:
    x.y = (x.y - mean) / std

random.shuffle(data_list)

criterion = nn.MSELoss()


avg_test_loss_gcn = numpy.empty(6)
avg_test_loss_gat = numpy.empty(6)


model = GCN(mconv.num_atm_feats, 1, readout='mean').cuda()
avg_test_loss_gcn[0], targets, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (mean): {:.4f}'.format(avg_test_loss_gcn[0]))
numpy.savetxt('pred_results/reg_result_mean.csv', numpy.hstack((targets, preds)), delimiter=',')

model = GCN(mconv.num_atm_feats, 1, readout='max').cuda()
avg_test_loss_gcn[1], targetrs, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (max): {:.4f}'.format(avg_test_loss_gcn[1]))
numpy.savetxt('pred_results/reg_result_max.csv', numpy.hstack((targets, preds)), delimiter=',')

model = GCN(mconv.num_atm_feats, 1, readout='attn').cuda()
avg_test_loss_gcn[2], targets, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (attn): {:.4f}'.format(avg_test_loss_gcn[2]))
numpy.savetxt('pred_results/reg_result_attn.csv', numpy.hstack((targets, preds)), delimiter=',')

model = GCN(mconv.num_atm_feats, 1, readout='lstm').cuda()
avg_test_loss_gcn[3], targets, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (lstm): {:.4f}'.format(avg_test_loss_gcn[3]))
numpy.savetxt('pred_results/reg_result_lstm.csv', numpy.hstack((targets, preds)), delimiter=',')

model = GCN(mconv.num_atm_feats, 1, readout='sum').cuda()
avg_test_loss_gcn[4], targets, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (add): {:.4f}'.format(avg_test_loss_gcn[4]))
numpy.savetxt('pred_results/reg_result_add.csv', numpy.hstack((targets, preds)), delimiter=',')

model = GCN(mconv.num_atm_feats, 1, readout='uattn').cuda()
avg_test_loss_gcn[5], targets, preds = tr.cross_validation(data_list, 5, model, criterion, num_epochs, batch_size, lr, 1e-5)
print('Test loss (un_attn): {:.4f}'.format(avg_test_loss_gcn[5]))
numpy.savetxt('pred_results/reg_result_uattn.csv', numpy.hstack((targets, preds)), delimiter=',')

print('------------------ GCN ------------------ ')
for i in range(0, 6):
    print(avg_test_loss_gcn[i])
