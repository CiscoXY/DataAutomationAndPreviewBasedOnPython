import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from  statsmodels.api import ProbPlot
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro,anderson , normaltest , jarque_bera
from scipy.stats import chi2
from Multivariate_statistical import Mahalanobis_Distance
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

def chi2_QQ(X , axes):
    '''
    绘制卡方QQ图(对应维数)
    传进来一个n , p维矩阵 , (不能是数据框)
    和要绘制的axes(plt的subplot,以及是否要正则化的bool值)
    '''
    temp = X
    if(X.ndim == 1):
        print('至少为2维才能绘制卡方图')
        exit(-1)
    n , p = X.shape
    chi2_plist = chi2.ppf((np.array(range(1,len(temp)+1))-0.5)/len(temp) , p)
    d = Mahalanobis_Distance(X)
    inf = np.min([d,chi2_plist]) ; sup = np.max([d,chi2_plist])
    x = np.arange(inf , sup + 0.1 , 0.1)
    axes.plot(x , x , color = 'green') #* 绘制y=x标准线
    axes.scatter(chi2_plist , np.sort(d) , s = 9 , alpha = 0.6)


if __name__ == '__main__':
    np.random.seed(111)
    x = np.random.normal(100 , 200 , 3000) + np.random.uniform(20,100 , 3000)
    plt.show()