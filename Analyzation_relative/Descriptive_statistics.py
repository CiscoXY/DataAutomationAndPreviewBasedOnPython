import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
    
def Two_dim_hist():
    print('NONE')


def One_dim_hist(data , figsize = (6,6) , bins = 10 , alpha = 0.7 , histtype = 'bar' , edgecolor = 'r'):
    fig, axes = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
    axes.hist(data , bins = bins , alpha = alpha , histtype = histtype , edgecolor = edgecolor)
    return fig , axes

if __name__ == '__main__':
    x = np.random.randint(100 , 200 , 100)
    fig , axes = One_dim_hist(x)
    plt.draw()