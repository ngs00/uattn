import numpy
import pandas
import matplotlib.pyplot as plt


data = numpy.array(pandas.read_csv('../pred_results/esol_weight/reg_weight_mean.csv', header=None))

plt.rcParams.update({'font.size': 18})
plt.xlabel('True')
plt.ylabel('Predicted')
plt.scatter(data[:, 0], data[:, 1], s=20, edgecolor='k')
plt.grid()
plt.savefig('../fig/weight.png', bbox_inches='tight', dpi=100)
plt.close()
