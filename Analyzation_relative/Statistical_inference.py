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

def Normality_test(data , kde = True , ax1 = None , ax2 = None):
    """
    用于正态性检验,返回三个值,fig、axes和统计量表

    Args:
        data (pd.Series ; list ; np.array): 输入的数据
        kde (bool, optional): 是否绘制核密度曲线. Defaults to True.
        ax1 (_type_, optional): plt.subplot的子图对象 , 用于绘制直方图. Defaults to None.
        ax2 (_type_, optional): plt.subplot的子图对象 , 用于绘制正态qq图 . Defaults to None.

    Returns:
        sta_frame(pd.DataFrame): 存储着正态性检验的统计量和对应p值的数据框
    """
    n = len(data)
    if(n >= 2000): # 大样本
        Lilliefors = list(lilliefors(data))
        Skewness_Kurtosis = normaltest(data) ; Skewness_Kurtosis = [Skewness_Kurtosis.statistic , Skewness_Kurtosis.pvalue]
        J_B = jarque_bera(data) ; J_B = [J_B.statistic , J_B.pvalue]
        sta_frame = pd.DataFrame(np.array([Lilliefors , Skewness_Kurtosis , J_B]) , columns = ['statistic' , 'p-value'] , index = ['Lilliefors' , 'Skewness_Kurtosis' , 'Jarque-Bera'])
        
    else: #  小样本
        Shapiro = shapiro(data) ; Shapiro = [Shapiro.statistic , Shapiro.pvalue]
        Anderson = anderson(data)
        Index = Anderson.critical_values>Anderson.statistic
        Index = [i for i, x in enumerate(Index) if (not x)]
        if(len(Index)):
            Anderson = [Anderson.statistic , Anderson.significance_level[max(Index)]/100] # 获得对应的统计量和p值
        else:
            Anderson = [Anderson.statistic , 0.15] # 如果index本身是空的，那么说明哪个显著性水平都不能说明数据非正态
        sta_frame = pd.DataFrame(np.array([Shapiro , Anderson]) , columns = ['statistic' , 'p-value'] , index = ['Shapiro-Wilk' , 'Anderson-Darling'])
        
    if(ax1 != None):
        sns.histplot(data , kde = kde , ax = ax1)
        ax1.set_title("Histogram")
    
    if (ax2 != None):
        qqplot = ProbPlot(data , fit = True)
        qqplot.qqplot(line = 'q' , ax = ax2)
        ax2.set_title("Norm QQ")
        
    return sta_frame
    
    
if __name__ == '__main__':
    np.random.seed(111)
    x = np.random.normal(100 , 200 , 3000) + np.random.uniform(20,100 , 3000)
    fig , axes , sta_frame = Normality_test(x , plot = True)
    print(sta_frame.info())
    plt.show()