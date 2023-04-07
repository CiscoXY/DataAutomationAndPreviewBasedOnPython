import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from  statsmodels.api import ProbPlot
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro,anderson , normaltest , jarque_bera
#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)


def One_dim_hist(data , figsize = (6,6) , bins = 10 , alpha = 0.7 , histtype = 'bar' , edgecolor = 'b'):
    fig, axes = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
    axes.hist(data , bins = bins , alpha = alpha , histtype = histtype , edgecolor = edgecolor)
    return fig , axes




if __name__ == '__main__':
    np.random.seed(111)
    x = np.random.normal(100 , 200 , 3000) + np.random.uniform(20,100 , 3000)
    plt.show()