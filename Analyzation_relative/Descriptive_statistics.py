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
#### 小工具
def foo(data):
    """
    一个用于判断传入数据是dataframe类型还是array类型
    """
    if isinstance(data, pd.DataFrame):
        return 1
    elif isinstance(data, np.ndarray):
        return 0

def DataLabeling(data):  # 给数据贴标签
    """
    data : Dataframe ; 
    
    用途： 
    给data贴标签,总计有3种标签, 1种异常
    数值型：对应标签'numeric',数值为0; 二分类型: 对应标签'Binary',数值为1; 多分类型(包括无序和有序型): 'Multivar',数值为2
    如果只有一种值,则返回异常,对应数值1
    并返回对应列的数据类型list(自动判断,有一定误差,更为精准还是人为贴标签合适）
    
    例如,传入数据为3维,第一列是数值型,第二列是二分类型,第三列是多分类型,那么
    return [0 , 1 , 2]
    
    判断依据：
    对于数值型数据,如果不同数值超过50个,则判断为数值型
    对于二分类型数据,如果只存在两个值,那么认定为2分类数据
    对于多分类数据,如果存在>=3 , <=50 , 则判断为分类型数据
    
    其中,值可以是数字,也可以是文本
    
    
    注： 自动识别在特殊情况下容易出现错误 , 如果可以建议手动和自动相结合.
    """
    assert isinstance(data, pd.DataFrame) , "不是 dataframe 请转换后输入"
    result = []
    for col in data.columns:
        n = len(data[col].value_counts())
        if n == 1:
            result.append(-1)
        elif n <= 2:
            result.append(1)
        elif n > 50:
            result.append(0)
        else:
            result.append(2)
    return result




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
    df = pd.read_csv("data/test_data.csv",encoding = "utf-8")
    Chaoyang = df.loc[df["region"] == "朝阳" , ['rent' , 'area' , 'room' , 'subway']]
    print(Chaoyang)
    print(DataLabeling(Chaoyang))