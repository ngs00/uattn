import numpy
import pandas
import matplotlib.pyplot as plt


def plot_reg_results(results, readout_name):
    max = numpy.max(results[:, 0]) + 0.5
    min = numpy.min(results[:, 0]) - 0.5
    plt.rcParams.update({'font.size': 28})
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.scatter(results[:, 0], results[:, 1], s=20, edgecolor='k')
    plt.grid()
    plt.plot([min, max], [min, max], 'k')
    plt.xlim([min, max])
    plt.ylim([min, max])
    plt.savefig('../fig/reg_result_' + readout_name + '.png', bbox_inches='tight', dpi=100)
    plt.close()


path = '../pred_results/freesolv/'
train_mean = numpy.array(pandas.read_csv(path + 'reg_result_mean.csv', header=None))
train_max = numpy.array(pandas.read_csv(path + 'reg_result_max.csv', header=None))
train_attn = numpy.array(pandas.read_csv(path + 'reg_result_attn.csv', header=None))
train_lstm = numpy.array(pandas.read_csv(path + 'reg_result_lstm.csv', header=None))
train_add = numpy.array(pandas.read_csv(path + 'reg_result_add.csv', header=None))
train_uattn = numpy.array(pandas.read_csv(path + 'reg_result_uattn.csv', header=None))

plot_reg_results(train_mean, 'mean')
plot_reg_results(train_max, 'max')
plot_reg_results(train_attn, 'attn')
plot_reg_results(train_lstm, 'lstm')
plot_reg_results(train_add, 'add')
plot_reg_results(train_uattn, 'uattn')
