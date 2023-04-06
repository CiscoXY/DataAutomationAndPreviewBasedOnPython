import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from  statsmodels.api import ProbPlot
import itertools   #* 用于进行解释变量名称的遍历。
import sys
#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)

def Formula_create(Label_list):
    """
    用于创建smf的formula,默认变量list的第一个元素为被解释变量
    后续所有的变量均为解释变量
    即创建所有子集对应的formula
    """
    k = len(Label_list)
    if k<2: 
        print("The length of Label isn't enough")
        sys.exit(1)
    # 创建解释变量的列表
    Variable_list = [str(x) for x in Label_list[1:]]
    Explained_variable = str(Label_list[0])
    # 创建一个空列表，用来存放结果
    result = []
    # 遍历每个可能的组合长度
    for n in range(1, k + 1):
        # 创建一个包含n个元素的所有组合的迭代器
        combinations = itertools.combinations(Variable_list, n)
        # 遍历每个组合
        for c in combinations:
            # 把每个元素转换成字符串，并且用加号连接起来
            s = "+".join(x for x in c)
            # 把字符串添加到结果列表中
            result.append("~".join([Explained_variable, s]))
    # 返回结果列表
    return result

def res_plot(Residual , figsize = (12 , 4) , dpi = 100):
    """
    该函数主要对残差进行散点图,pp、qq图绘制
    并返回matplotlib的fig和axes
    """
    fig, axes = plt.subplots(1,3 , figsize=figsize, dpi=dpi)
    X = np.arange(1,len(Residual)+1)
    axes[0].scatter(X , Residual , s = 2 , alpha = 0.4)
    axes[0].axhline(y = np.mean(Residual) , color='r', linestyle='--')
    pqplot = ProbPlot(Residual , fit = True)
    ppplot = pqplot.ppplot(line = '45' , ax = axes[1])
    qqplot = pqplot.qqplot(line = 'q' , ax = axes[2])
    axes[0].set_title("Scatter of res")
    axes[1].set_title('Normal PP plot')
    axes[2].set_title('Normal QQ plot')
    return fig , axes




if __name__=="__main__":
    x = np.random.uniform(100 , 200 , 100)
    fig , axes = res_plot(x)
    plt.show()