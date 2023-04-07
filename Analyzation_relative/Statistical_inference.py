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

def Normality_test(data , figsize = (8 , 4) , dpi = 100 , kde = True , plot = True):
    """
    用于正态性检验,返回三个值,fig、axes和统计量表
    plot = True 默认为画图
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
        Anderson = [Anderson.statistic , Anderson.significance_level[max(Index)]/100] # 获得对应的统计量和p值
        sta_frame = pd.DataFrame(np.array([Shapiro , Anderson]) , columns = ['statistic' , 'p-value'] , index = ['Shapiro-Wilk' , 'Anderson-Darling'])
        
    if(plot):
        fig , axes = plt.subplots(1 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white")
        sns.histplot(data , kde = kde , ax = axes[0])
        qqplot = ProbPlot(data , fit = True)
        qqplot.qqplot(line = 'q' , ax = axes[1])
        axes[0].set_title("Histogram")
        axes[1].set_title("Norm QQ")
        return fig , axes , sta_frame
    else:
        return sta_frame
    
    
if __name__ == '__main__':
    np.random.seed(111)
    x = np.random.normal(100 , 200 , 3000) + np.random.uniform(20,100 , 3000)
    fig , axes , sta_frame = Normality_test(x , plot = True)
    print(sta_frame.info())
    plt.show()