import numpy
import copy
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


def cross_validation(data_list, k, default_model, criterion, num_epochs, batch_size, lr, l2, reg=True):
    folds = get_k_fold(data_list, k)
    list_targets = list()
    list_preds = list()
    test_vals = 0
    test_accs = 0
    test_f1s = 0

    for i in range(0, k):
        print('--------- Fold {} ---------'.format(i + 1))

        train_dataset, test_dataset = get_train_test(folds, i)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        model = copy.deepcopy(default_model)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

        for epoch in range(0, num_epochs):
            if reg:
                train_loss = train(model, optimizer, train_data_loader, criterion)
                test_loss, _, _ = test(model, test_data_loader, criterion)
                print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest loss: {:.4f}'
                      .format(epoch + 1, num_epochs, train_loss, test_loss))
            else:
                # train_loss = train_clf(model, optimizer, train_data_loader, criterion)
                # test_acc, test_f1, test_matthews = test_clf(model, test_data_loader)
                train_loss = train_bench(model, optimizer, train_data_loader, criterion)
                test_f1 = test_bench(model, test_data_loader)
                # print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest acc: {:.4f}\tTest F1: {:.4f}\tTest M. coeff: {:.4f}'
                #       .format(epoch + 1, num_epochs, train_loss, test_acc, test_f1, test_matthews))
                print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest F1: {:.4f}'
                      .format(epoch + 1, num_epochs, train_loss, test_f1))

        if reg:
            test_val, targets, preds = test(model, test_data_loader, criterion)
            test_vals += test_val
            list_targets.append(targets)
            list_preds.append(preds)
        else:
            # test_acc, test_f1, test_matthews = test_clf(model, test_data_loader)
            test_f1 = test_bench(model, test_data_loader)
            # test_accs += test_acc
            test_f1s += test_f1

    if reg:
        return test_vals / k, torch.cat(list_targets, dim=0).cpu().numpy(), torch.cat(list_preds, dim=0).cpu().numpy()
    else:
        return test_accs / k, test_f1s / k


def get_k_fold(data_list, k):
    size_fold = int(len(data_list) / k)
    folds = list()

    for i in range(0, k):
        if i == k - 1:
            folds.append(data_list[size_fold*i:])
        else:
            folds.append(data_list[size_fold*i:size_fold*(i+1)])

    return folds


def get_train_test(folds, k):
    train_data_list = list()

    for i in range(0, len(folds)):
        if i != k:
            train_data_list += folds[i]

    return train_data_list, folds[k]


def train(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch.batch = batch.batch.cuda()

        pred = model(batch)
        loss = criterion(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

        if (i + 1) % 20 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return numpy.sqrt(train_loss / len(data_loader))


def train_clf(model, optimizer, data_loader, criterion):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        batch.batch = batch.batch.cuda()

        pred = model(batch)
        loss = criterion(pred, batch.y.long().squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()

        if (i + 1) % 200 == 0:
            print('[' + str(i + 1) + '/' + str(len(data_loader)) + ']')

    return train_loss / len(data_loader)


def test(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    targets = list()
    preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            pred = model(batch)
            test_loss += criterion(batch.y, pred).detach().item()
            targets.append(batch.y)
            preds.append(pred)

    return numpy.sqrt(test_loss / (len(data_loader))), torch.cat(targets, dim=0), torch.cat(preds, dim=0)


def test_clf(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    pred_result = list()

    f1_pred = list()
    f1_target = list()

    with torch.no_grad():
        for batch in data_loader:
            batch.batch = batch.batch.cuda()

            pred = model(batch)
            _, predicted = torch.max(pred, 1)
            correct += (predicted.view(-1, 1) == batch.y.long()).sum().item()
            total += batch.y.shape[0]
            pred_result.append(pred)

            f1_pred.append(predicted.cpu().numpy())
            f1_target.append(batch.y.squeeze(1).cpu().numpy())

    accuracy = 100 * (correct / float(total))

    f1_pred = numpy.hstack(f1_pred)
    f1_target = numpy.hstack(f1_target)

    return accuracy, f1_score(f1_pred, f1_target, average='weighted'), matthews_corrcoef(f1_pred, f1_target)


def train_bench(model, optimizer, train_loader, criterion):
    model.train()

    total_loss = 0

    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = data.batch.cuda()
        targets = data.y.squeeze(1)

        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, targets)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


def test_bench(model, data_loader):
    model.eval()

    ys, preds = [], []

    for data in data_loader:
        data.batch = data.batch.cuda()
        ys.append(data.y)
        with torch.no_grad():
            out = model(data)
        preds.append((out > 0).float().cpu())

    y = ys[0].squeeze(1).cpu().numpy()
    _, ind = torch.max(preds[0], dim=1)
    pred = ind.cpu().numpy()

    # return f1_score(y, pred, average='micro')
    return numpy.sum(pred == y) / len(preds)
