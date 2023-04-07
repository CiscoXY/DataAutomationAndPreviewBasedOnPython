import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from  statsmodels.api import ProbPlot
import itertools   #* 用于进行解释变量名称的遍历。
import sys
from scipy.stats import spearmanr
from Statistical_inference import Normality_test
from statsmodels.stats.diagnostic import het_goldfeldquandt
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
    fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
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

def res_test(Residual , fittedvalues , X , significance_value = 0.05):
    """
    X : 数据阵,默认第一列为被解释变量
    significance_value默认为0.05
    包含三部分,正态性test,异方差test,自相关test
    """
    # 正态性test
    norm_frame = Normality_test(Residual, plot=False) 
    if((norm_frame['p-value']<significance_value).all()):
        print('The p-values show that the residual is strongly Non-normal')
    elif((norm_frame['p-value']<significance_value).any()):
        print('The p-values show that the residual may be Non-normal')
    else:
        print('The p-values show that the residual is likely normal distributed')

    # * Part2 异方差部分
    # 获取表头list
    Labels = X.columns[1:]
    
    # spearman秩相关test
    res_abs = np.abs(Residual)
    spearman_df = pd.DataFrame(data=None,columns=['corr','p-value']) # 创建空表
    for label in Labels:
        corr , p_value = spearmanr(X[label] , b = res_abs)
        if(p_value<= significance_value):
            # 如果出现spearman秩相关test显著，则添加相应数据到对应的dataframe当中
            print('The p-value shows it is statistical significant that '+ label + ' is spearman_corr with the res')
            spearman_df.loc[label] = [corr , p_value]
    
    # Goldfeld-Quandt检验
    Goldfeld_Quandt_df = pd.DataFrame(data = None , columns = ['F statistic' , 'p-value'])
    for index , label in enumerate(Labels):
        [F , p_value , order] = het_goldfeldquandt(X.iloc[:,0] , X[Labels] , idx = index , split = 0.4) # 取前后40%的数据作为两个组进行格登菲尔德检验
        if(p_value<= significance_value):
            # 如果出现Goldfeld_Quandt显著，则添加相应数据到对应的dataframe当中
            print('The G_Q test shows that '+label +' is the cause of heteroscedasticity')
            Goldfeld_Quandt_df.loc[label] = [F , p_value]
    
    # Glejser 检验
    
    
    
    return norm_frame , spearman_df , Goldfeld_Quandt_df


if __name__=="__main__":
    df = pd.read_csv("data/test_data.csv",encoding = "utf-8")
    Chaoyang = df.loc[df["region"] == "朝阳" , ['rent' , 'area' , 'room' , 'subway']]
    model1_CY = smf.ols('rent ~ area + room + subway' , data =Chaoyang).fit()
    fitvalue1_CY = model1_CY.fittedvalues
    res = model1_CY.resid
    df1 , df2 , df3 = res_test(res , fitvalue1_CY , Chaoyang)
    print(df1 ,'\n', df2 , '\n' , df3)
    