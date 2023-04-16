import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools   #* 用于进行解释变量名称的遍历。
import sys
import statsmodels.formula.api as smf
from  statsmodels.api import ProbPlot
from scipy.stats import spearmanr
from Statistical_inference import Normality_test
from statsmodels.stats.diagnostic import het_goldfeldquandt , het_breuschpagan , het_white , acorr_ljungbox
from statsmodels.stats.outliers_influence import reset_ramsey , variance_inflation_factor
from sklearn.linear_model import Ridge
#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)

def foo(data):
    """
    一个用于判断传入数据是dataframe类型还是array类型
    """
    if isinstance(data, pd.DataFrame):
        return 1
    elif isinstance(data, np.ndarray):
        return 0

def Formula_create(Label_list):
    """创建自变量和因变量的所有可能组合的公式"""
    assert len(Label_list) >= 2, "The length of Label isn't enough"
    Variable_list = Label_list[1:] 
    result = []
    for n in range(1, len(Variable_list) + 1): 
        for c in itertools.combinations(Variable_list, n):
            s = "+".join(c)
            result.append("~".join([Label_list[0],s]))  
    return result

def Formula_encoder(Label_list):
    """
    根据给的list创建公式
    """
    assert len(Label_list) >= 2, "The length of Label isn't enough"
    result = Label_list[0]+'~'+Label_list[1]
    if(len(Label_list) > 2):
        result += '+'
        result += "+".join(Label_list[2:])
    return result

def Formula_decoder(string):
    """
    根据公式返还对应的Label_list
    """
    Label_list = []
    word = ''
    for c in string:
        if c == '~' or c == '+':
            Label_list.append(word)
            word = ''
        else:
            word += c
    Label_list.append(word)
    return Label_list

    
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


def res_test(Residual , X , significance_value = 0.05):
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

def Multicollinearity_test(data):
    """
    用于检验多重共线性,data应当全是解释变量组成的dataframe或者np.array
    返回的dataframe由两部分组成
    方差扩大因子法和特征值（条件指数法）`
    """
    Mul_df = pd.DataFrame()
    
    if isinstance(data, pd.DataFrame):
        Columns = data.columns
        X = data.values
    else: 
        X = data
        Columns = [f'x{i+1}' for i in range(X.shape[1])]
        
    # 方差扩大因子法
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    # 条件指数
    XTX = X.T @ X 
    eigenvalues = np.linalg.eigvals(XTX)
    max_eig = np.max(eigenvalues)  
    min_eig = np.min(eigenvalues)
    con_index = np.sqrt(max_eig/min_eig)
    
    index = [f'{col}_VIF' for col in Columns] + ['条件指数']
    Mul_df = pd.DataFrame({'value': vif + [con_index]}, index=index)
    
    return Mul_df

def Ridge_trace_analysis(data , explained_var = None , k = np.arange(0 , 100 , 1) , ax = None):
    """
    传入dataframe类型的data,默认第一列为被解释变量，如果需要指定被解释变量，输入对应的列名称即可，为字符串
    该函数主要进行岭回归,返回一个多维np矩阵,第一列为取的k的范围,默认为0-10,步长为0.1,剩下各列为回归系数在不同k取值下的岭回归值。
    ax默认为None,即不绘图,如果传入ax则在这个ax上根据该np多维矩阵绘制岭迹分析图
    """
    # 获取被解释变量列名,默认取第一列
    if explained_var is None:
        explained_var = data.columns[0]
        
    # 指定被解释变量和自变量 
    y = data[explained_var]
    X = data.drop([explained_var], axis=1)
    
    # 岭回归,计算各alpha下的回归系数
    coef_arr = []
    for alpha in k:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coef_arr.append(ridge.coef_)
    # 构造返回值   
    ret = np.array(k).reshape(-1, 1)
    ret = np.hstack((ret, np.array(coef_arr)))
    
    if ax is not None:  
        for i, coef in enumerate(ret.T[1:]): 
            ax.plot(k, coef, lw=3, label=data.columns[i]) 
            
        ax.set_xlabel('k')
        ax.set_ylabel('value')
        ax.set_title('岭迹分析图')
        ax.legend()
    return ret
    
if __name__=="__main__":
    df = pd.read_csv("data/test_data.csv",encoding = "utf-8")
    Chaoyang = df.loc[df["region"] == "朝阳" , ['rent' , 'area' , 'room' , 'subway']]
    # model1_CY = smf.ols('rent ~ area + room + subway' , data =Chaoyang).fit()
    # fitvalue1_CY = model1_CY.fittedvalues
    # res = model1_CY.resid
    
    # df1 , df2 , df3 , df4  , df5= res_test(res , fitvalue1_CY , Chaoyang)
    # print(df1 ,'\n', df2 , '\n' , df3 , '\n' , df4  , '\n' , df5)
    
    
    # fig , axes = res_plot(res , fitvalue1_CY)
    # plt.show()
    
    # print(Endogeneity_test(model1_CY))
    
    #print(Multicollinearity_test(Chaoyang[['area' , 'room' , 'subway']]))
    
    # fig , axes = plt.subplots(1 , 1 , figsize = (8 , 8) , dpi = 100)
    # print(Ridge_trace_analysis(Chaoyang ,k = np.arange(0 , 10000 , 10) ,  ax = axes))
    # plt.show()
    print(Formula_decoder(Formula_encoder(Chaoyang.columns)))