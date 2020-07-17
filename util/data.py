import numpy
import pandas
import torch
from sklearn import preprocessing
from torch_geometric.data import Data


def load_data(dataset_name):
    data_list = list()
    path = '../data/' + dataset_name
    adj = numpy.array(pandas.read_csv(path + '/' + dataset_name + '_A.txt', header=None), dtype=numpy.int)
    grp_ind = numpy.array(pandas.read_csv(path + '/' + dataset_name + '_graph_indicator.txt', header=None), dtype=numpy.int)
    node_feats = numpy.array(pandas.read_csv(path + '/' + dataset_name + '_node_attributes.txt', header=None), dtype=numpy.float)
    labels = numpy.array(pandas.read_csv(path + '/' + dataset_name + '_graph_labels.txt', header=None), dtype=numpy.long) - 1

    node_feats = preprocessing.scale(node_feats)
    num_nodes = get_num_nodes(grp_ind)
    edges = get_edges(adj, num_nodes)
    node_feat_mat = get_node_feat_mat(node_feats, num_nodes)

    for i in range(0, len(num_nodes)):
        grp = Data(x=torch.tensor(node_feat_mat[i], dtype=torch.float).cuda(),
                   edge_index=edges[i].t().contiguous(),
                   y=torch.tensor(labels[i], dtype=torch.long).view(-1, 1).cuda(), idx=i)
        data_list.append(grp)

        if (i + 1) % 50 == 0:
            print('Loading ' + str(i + 1) + 'th graph data was completed.')

    return data_list, node_feats.shape[1]


def get_num_nodes(grp_ind):
    num_nodes = list()
    ind = grp_ind[0]
    num = 1

    for i in range(1, grp_ind.shape[0]):
        if ind == grp_ind[i]:
            num += 1
        else:
            num_nodes.append(num)
            ind = grp_ind[i]
            num = 1
    num_nodes.append(num)

    return num_nodes


def get_edges(adj, num_nodes):
    edges = list()
    last_ind = 0
    pos = 0
    fct_id = 1

    for i in range(0, len(num_nodes)):
        edges.append(list())
        last_ind += num_nodes[i]

        if num_nodes[i] == 1:
            edges[i].append((0, 0))
        else:
            num = 0
            for j in range(pos, adj.shape[0]):
                if adj[j, 0] <= last_ind and adj[j, 0] <= last_ind:
                    edges[i].append((adj[j, 0] - fct_id, adj[j, 1] - fct_id))
                    num += 1
                else:
                    pos += num
                    break

        edges[i] = torch.tensor(edges[i], dtype=torch.long).cuda()
        fct_id += num_nodes[i]

    return edges


def get_node_feat_mat(node_feats, num_nodes):
    feats = list()
    pos = 0

    for i in range(0, len(num_nodes)):
        feats.append(node_feats[pos:pos+num_nodes[i], :])
        pos += num_nodes[i]

    return feats
