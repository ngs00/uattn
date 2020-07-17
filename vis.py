import numpy
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import util.mol_conv as mconv
from model.GCN import GCN
from model.GIN import GIN


def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch.batch = batch.batch.cuda()

        pred, _ = model(batch)
        loss = criterion(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return numpy.sqrt(train_loss / len(data_loader))


def emb(model, data_loader):
    model.eval()
    embs = list()
    targets = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            _, emb = model(batch)
            embs.append(emb)
            targets.append(batch.y)

    return torch.cat(embs, dim=0), torch.cat(targets, dim=0)


lr = 5e-5
l2 = 1e-5


data_list = mconv.read_dataset('data/esol.csv')
# random.shuffle(data_list)
# train_data_list = data_list[:int(len(data_list) * 0.8)]
# test_data_list = data_list[int(len(data_list) * 0.8):]

data = DataLoader(data_list, batch_size=128)
# train_data = DataLoader(train_data_list, batch_size=128, shuffle=True)
# test_data = DataLoader(test_data_list, batch_size=128)
gcn = GCN(mconv.num_atm_feats, 1, readout='sum').cuda()
# gcn = GIN(mconv.num_atm_feats, 1, readout='sum').cuda()
opt_gcn = optim.Adam(gcn.parameters(), lr=lr, weight_decay=l2)
criterion = nn.MSELoss()

for i in range(0, 1):
    train_loss = train(gcn, opt_gcn, data, criterion)
    print(i, train_loss)

emb, targets = emb(gcn, data)
# mol_weights = scale(numpy.array([x.mol_weight for x in data_list])).reshape(-1, 1)
# emb = numpy.hstack([emb.cpu().numpy(), mol_weights])
emb = emb.cpu().numpy()
targets = targets.cpu().numpy()

emb_gcn_tsne = TSNE(n_components=2).fit_transform(emb)
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.xlabel('$h_1$')
plt.ylabel('$h_2$')
plt.scatter(emb_gcn_tsne[:, 0], emb_gcn_tsne[:, 1], c=targets, edgecolors='k')
plt.colorbar()
plt.savefig('esol_emb_rand_sum.png', bbox_inches='tight', dpi=100)
plt.close()
