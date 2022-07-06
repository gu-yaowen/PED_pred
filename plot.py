import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def scatter_plot(x_test, y_test, pre, value, value_type, value2, value_type2):
    index_0 = np.where(x_test[:, -2] == 0)
    index_1 = np.where(x_test[:, -2] == 1)
    index_2 = np.where(x_test[:, -2] == 2)
    y_test_0 = y_test[index_0]
    y_test_1 = y_test[index_1]
    y_test_2 = y_test[index_2]
    pre_0 = pre[index_0]
    pre_1 = pre[index_1]
    pre_2 = pre[index_2]
    plt.figure(figsize=(6, 6), dpi=300)
    ax = plt.axes(facecolor='#E6E6E6')
    ax.set_axisbelow(True)
    plt.grid(color='w', linestyle='solid')
    for spine in ax.spines.values():
        spine.set_visible(False)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
    ax.plot([0, 500], [0, 500],
            c='gray', label=value_type + ' score =%0.3f' % value + '\n' + value_type2 + '=%.3f' % value2)
    ax.scatter(y_test_0, pre_0, s=1, label='green', alpha=0.5)
    ax.scatter(y_test_1, pre_1, s=1, label='yellow', alpha=0.5)
    ax.scatter(y_test_2, pre_2, s=1, label='red', alpha=0.5)
    plt.legend(loc="lower right")
    plt.xlabel('True label')
    plt.ylabel('Predict label')
    plt.xlim(0, 500, 0.1)
    plt.ylim(0, 500, 0.1)
