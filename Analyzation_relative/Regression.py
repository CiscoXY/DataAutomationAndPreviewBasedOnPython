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
from statsmodels.stats.diagnostic import het_goldfeldquandt , het_breuschpagan , het_white , acorr_ljungbox
from statsmodels.stats.outliers_influence import reset_ramsey
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

def res_plot(Residual ,fittedvalues , figsize = (8 , 8) , dpi = 100):
    """
    该函数主要对残差进行散点图,时序图
    pp、qq图绘制
    并返回matplotlib的fig和axes
    """
    fig, axes = plt.subplots(2,2 , figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
    X = np.arange(1,len(Residual)+1)
    axes[0][0].scatter(fittedvalues , Residual , s = 2 , alpha = 0.4 , color = 'blue')
    axes[0][1].scatter(X , Residual , s = 2 , alpha = 0.4)
    axes[0][1].axhline(y = np.mean(Residual) , color='r', linestyle='--')
    pqplot = ProbPlot(Residual , fit = True)
    ppplot = pqplot.ppplot(line = '45' , ax = axes[1][0])
    qqplot = pqplot.qqplot(line = 'q' , ax = axes[1][1])
    axes[0][0].set_title('Scatter of res and y_predict')
    axes[0][0].set_xlabel('y_predict') ; axes[0][0].set_ylabel('Residuals')
    axes[0][1].set_title("Scatter of res")
    axes[1][0].set_title('Normal PP plot')
    axes[1][1].set_title('Normal QQ plot')
    plt.subplots_adjust(hspace=0.3)
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
            spearman_df.loc[label] = [corr , p_value]
    
    # Goldfeld-Quandt检验
    Goldfeld_Quandt_df = pd.DataFrame(data = None , columns = ['F statistic' , 'p-value'])
    for index , label in enumerate(Labels):
        [F , p_value , order] = het_goldfeldquandt(X.iloc[:,0] , X[Labels] , idx = index , split = 0.4) # 取前后40%的数据作为两个组进行格登菲尔德检验
        if(p_value<= significance_value):
            # 如果出现Goldfeld_Quandt显著，则添加相应数据到对应的dataframe当中
            Goldfeld_Quandt_df.loc[label] = [F , p_value]
    
    # 接下来是基于回归的异方差检验
    Reg_relative_df =  pd.DataFrame(data=None,columns=['statistic','p-value']) # 创建空表
    # Breusch-Pagan 检验
    [lm , lm_pvalue , F , F_pvalue] = het_breuschpagan(Residual , X[Labels])
    Reg_relative_df.loc['BP_LM'] = [lm , lm_pvalue]
    Reg_relative_df.loc['BP_F'] = [F , F_pvalue]
    # White 检验
    for col in Labels:
        # 获取该列的唯一值
        unique_values = X[col].unique()
        # 如果该列的唯一值只有两个，并且都是0或1，说明该列是二元的
        if len(unique_values) == 2 and all(x in [0, 1] for x in unique_values):
            judge = True
    if(not judge): # 如果不是二元且均为0和1，则可以进行white检验
        [lm , lm_pvalue , F , F_pvalue] = het_white(Residual , X[Labels])
        Reg_relative_df.loc['White_LM'] = [lm , lm_pvalue]
        Reg_relative_df.loc['White_F'] = [F , F_pvalue]
    
    # 使用时序对于纯随机序列的LB统计量进行检验  自相关test
    LB_df = pd.DataFrame(np.array(acorr_ljungbox(res, lags=10)).T , columns = ['LB_statistic' , 'p-value'] , index = np.arange(1,11))
    
    return norm_frame , spearman_df , Goldfeld_Quandt_df , Reg_relative_df , LB_df

def Endogeneity_test(model , degree = 5):
    """
    用于检验内生性,返两个test的统计量值和对应的p值
    test是RESET 即 regression specification error test 用于检验模型是否因为遗漏高次项导致出现内生性
    """
    End_df = pd.DataFrame(data = None , columns = ['statistic' , 'p-value'])
    result = reset_ramsey(res=model, degree=degree)
    End_df.loc['RESET_test'] = [result.fvalue[0][0] , result.pvalue]
    return End_df

def Multicollinearity_test(data , ):
    """
    用于检验多重共线性
    返回的dataframe由两部分组成
    方差扩大因子法和特征值（病态指数法）`
    """
    Mul_df = pd.DataFrame(data = None , columns = ['statistic' , 'value'])
    
    
    
    
    
    

if __name__=="__main__":
    df = pd.read_csv("data/test_data.csv",encoding = "utf-8")
    Chaoyang = df.loc[df["region"] == "朝阳" , ['rent' , 'area' , 'room' , 'subway']]
    model1_CY = smf.ols('rent ~ area + room + subway' , data =Chaoyang).fit()
    fitvalue1_CY = model1_CY.fittedvalues
    res = model1_CY.resid
    # df1 , df2 , df3 , df4  , df5= res_test(res , fitvalue1_CY , Chaoyang)
    # print(df1 ,'\n', df2 , '\n' , df3 , '\n' , df4  , '\n' , df5)
    # fig , axes = res_plot(res , fitvalue1_CY)
    # plt.show()
    print(Endogeneity_test(model1_CY))