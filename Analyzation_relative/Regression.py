import sys
sys.path.append('./')
from datetime import datetime
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools   #* 用于进行解释变量名称的遍历。
import random

import statsmodels.formula.api as smf
from  statsmodels.api import ProbPlot
from scipy.stats import spearmanr

from Analyzation_relative.Statistical_inference import Normality_test

from statsmodels.stats.diagnostic import het_goldfeldquandt , het_breuschpagan , het_white , acorr_ljungbox
from statsmodels.stats.outliers_influence import reset_ramsey , variance_inflation_factor
from sklearn.linear_model import Ridge


#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)
# 小工具

def Formula_create(Label_list):
    """创建自变量和因变量的所有可能组合的公式 , 其中Label_list的第一个元素默认为被解释变量"""
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

def random_subset(labels , k = 3):
    """
    从labels当中随机抽取k个label,
    返回这k个label组成的list和剩余labels组成的list
    k默认取值为3
    """
    n = len(labels)
    idx = random.sample(range(n), k)  # 随机选择3个索引
    subset = [labels[i] for i in idx]  # 根据索引获取子列表
    remaining = [labels[i] for i in range(n) if i not in idx]  # 获取剩余列表
    
    return subset, remaining

### 各种Test相关



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


def res_test(Residual , X , significance_value = 0.05 , higher_term = True , labels = None):
    """
    X : 数据阵,默认第一列为被解释变量
    significance_value默认为0.05
    包含三部分,正态性test,异方差test,自相关test
    
    Labels : 是否指定参与模型的自变量名称,应为list型,如果不指定,则默认所有变量参与了建模
    """
    # 正态性test
    norm_df = Normality_test(Residual, plot=False) 
    # if((norm_frame['p-value']<significance_value).all()):
    #     print('The p-values show that the residual is strongly Non-normal')
    # elif((norm_frame['p-value']<significance_value).any()):
    #     print('The p-values show that the residual may be Non-normal')
    # else:
    #     print('The p-values show that the residual is likely normal distributed')

    
    # 获取表头list
    if(labels == None):
        Labels = X.columns[1:]
    else:
        Labels = labels
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
    judge = False
    for col in Labels:
        # 获取该列的唯一值
        unique_values = X[col].unique()
        # 如果该列的唯一值只有两个，并且都是0或1，说明该列是二元的
        if len(unique_values) == 2 and all(x in [0, 1] for x in unique_values):
            judge = True
    if((not judge) and len(Labels)<= 50 and (not higher_term)): # 如果不是二元且均为0和1且数据中不含有高次项，则可以进行white检验
        [lm , lm_pvalue , F , F_pvalue] = het_white(Residual , X[Labels])
        Reg_relative_df.loc['White_LM'] = [lm , lm_pvalue]
        Reg_relative_df.loc['White_F'] = [F , F_pvalue]
    
    # 使用时序对于纯随机序列的LB统计量进行检验  自相关test
    LB_df = pd.DataFrame(np.array(acorr_ljungbox(Residual, lags=10 , return_df=False)).T , columns = ['LB_statistic' , 'p-value'] , index = np.arange(1,11))
    
    return norm_df , spearman_df , Goldfeld_Quandt_df , Reg_relative_df , LB_df

def Endogeneity_test(model , degree = 3):
    """
    用于检验内生性,返两个test的统计量值和对应的p值
    test是RESET 即 regression specification error test 用于检验模型是否因为遗漏高次项导致出现内生性
    """
    End_df = pd.DataFrame(data = None , columns = ['statistic' , 'p-value'])
    result = reset_ramsey(res=model, degree=degree)
    End_df.loc['RESET_test'] = [result.fvalue[0][0] , result.pvalue]
    return End_df

# 综合性检验，综合了正态性，异方差，自相关，内生性
def Model_exception_test(res , data , model , significance_level = 0.05 , higher_term = True , labels = None):
    """
    该函数旨在对模型进行综合性检验,包括正态性检验,异方差检验,自相关检验,内生性检验
    返回值包含两个,一个bool值,即模型是否通过了全部的检验;以及一个长度为4的list,内部值为对应是否通过检验
    例如如果一个模型通过了正态性,异方差,自相关检验,但是没有通过内生性检验,那么返回值为 False , [True , True , True , False] , 以及对应的显著的label组成的list
    
    res : 模型残差
    
    data : 原始数据,默认第一列为被解释变量,如果第一列不是被解释变量则会报错。
    
    model : 局外进行拟合后的模型
    
    significant_level : 显著性水平,如果检验的p值低于这个水平,则认为结果显著,不通过检验。
    
    higher_term : 是否含有高次项, 如果model本身就包含了解释变量的高次项,则应为True,否则在内生性检验和white检验中会出现无法计算矩阵逆的情况
    
    Labels : 是否指定参与模型的自变量名称,应为list型,如果不指定,则默认所有变量参与了建模
    """
    norm_df , spearman_df , GQ_df , Reg_relative_df , LB_df = res_test(res , data ,labels = labels ,  significance_value = significance_level,  higher_term=higher_term) # 获得正态性，异方差，自相关对应的dataframe
    if (not higher_term):
        End_df = Endogeneity_test(model) # 获得内生性test的dataframe
        End_bool = End_df['p-value'].gt(significance_level).all() # 判断内生性，尤其是因为遗漏高阶变量导致的内生性是否显著
    else : 
        End_bool = True # 如果不需要检验，则意味着模型中已经含有高次项，所以不会出现因为遗漏高阶变量导致的内生性出现。
    Norm_bool = norm_df['p-value'].gt(significance_level).all() # 判断正态性的两个检验是否都通过了，如果小于显著性水平则说明认为非正态
    Heter_bool = spearman_df.empty and GQ_df.empty and Reg_relative_df['p-value'].gt(significance_level).all() # 如果前两个空，最后一个表的所有p值均大于显著性水平，则认为通过了异方差检验
    Autocorr_bool = LB_df['p-value'].gt(significance_level).all() # 判断LB统计量是否都是大于0.05的，如果均是则认为不存在自相关
    Test_list = [Norm_bool , Heter_bool , Autocorr_bool , End_bool]
    
    if not spearman_df.empty:
        spearman_label = spearman_df.index.tolist()
    else:
        spearman_label = []
    if not GQ_df.empty:
        GQ_label = GQ_df.index.tolist()
    else:
        GQ_label = []
    
    return Norm_bool and Heter_bool and Autocorr_bool and End_bool , Test_list , spearman_label , GQ_label# 如果四个检验都通过了，返回True和对应的list，如果任意一个没有通过，则返回False , 和对应的list
    
    
# 多重共线性test,
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

#### 逐步回归(可以基于AIC，BIC，)
    # 注： 以下的逐步回归会优先考虑模型的假定问题，正态性，异方差性，自相关性以及内生性假定问题，随后才是参数的显著性问题。
def Stepwise_reg(data ,summary_output = True ,  significance_level = 0.05 , higher_term = True):
    """
    该函数主要对数据data进行逐步回归选择,优先考虑模型的各种异常情况，
    
    data : 数据阵,默认第一列是被解释变量
    summary_output : 是否将筛选的模型的summary以txt文件格式输出,默认为True,输出文件的目录为调用该模块的main.py函数的同一目录下的'result_output'文件夹内
    significance_level : 判断是否通过检验的显著性水平
    higher_term : 数据是否含有高维项,如果有则在检验时会忽略white检验和内生性检验
    
    return :dataframe   第1列是该formula是否通过了异常情况检验
                        第2,3,4,5列分别是正态性,异方差,自相关,内生性的检验通过与否的bool值
                        第6列存储着spearman秩相关检验下引起异方差的自变量名称,一个list,如果没有则为空
                        第7列存储着Goldfeld-Quandt检验下引起异方差的自变量名称,一个list,如果没有则为空
                        第8列存储着该formula对应的各种参数的t检验p值
                        第9列是该formula的调整的R方
                        第10列是F统计量的p值
                        第11列是formula,也就是对应的方程表达式,一船string
    """
    # 获得系统时间
    time = datetime.now()
    
    current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
    
    dir_path = './reg_result_output/'+current_time
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 初始化空表
    stepwise_columns = ['Pass_alltest' , 'Norm_test' , 'Heter_test' , 
                                                    'Autocorr_test' , 'End_test' , 
                                                    'spearman_label' , 'GQ_label' , 
                                                    'coef_pvalue' , 'adjust_R2' , 'F_pvalue' , 'formula']
    stepwise_df = pd.DataFrame(data=None,columns=stepwise_columns)
    # 初始随机选取3个label
    explained_vari = data.columns[0] # 获取被解释变量的label
    Labels = data.columns[1:].tolist() # 获取解释变量的labels
    for i in range(int(len(Labels) * 0.6)): # 进行极限30次的选取，如果30次都没有选择到不存在异方差和自相关的初始化label
        init_labels , left_labels = random_subset(Labels , k = 3)
        formula = Formula_encoder([explained_vari] + init_labels)
        print('init_formula is '+ formula)
        model = smf.ols(formula , data = data).fit()
        Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model ,labels = init_labels , higher_term = higher_term)
        if Bool_list[1] and Bool_list[2]:
            # 如果不存在 异方差和自相关 则初始化完成，像最终数据框中添加数据并跳出循环
            stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                        spearman_labels , GQ_labels , 
                                                        np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                ignore_index=True)
            if summary_output: # 如果需要输出则输出
                with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w') as f:
                    f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
            
            break
        else:
            init_labels = []
    if(len(init_labels) == 0) : #如果初始化labels空，则说明数据和模型本身存在极大异常，需要重新处理数据或者换模型。此时不优先考虑模型的基本假定相关检验（异方差，自相关，内生性等）
                                #反而根据参数的显著性，从全模型开始进行后退法筛选变量。
                                #基本过程为，全模型 -> 参数显著性检验 -> 随机剔除一个不显著的参数（如果没有不显著的参数了就停止迭代）-> 回归后做基本假定相关test并记录 -> 下一次迭代。
        while True: #开始后退法的循环
        
            formula = Formula_encoder([explained_vari] + Labels) # 依照Labels模型的方程式
            model = smf.ols(formula , data = data).fit()
            Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model ,labels = Labels , higher_term = higher_term)
        
            stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                        spearman_labels , GQ_labels , 
                                                        np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                ignore_index=True)
            if summary_output: # 如果需要输出则输出
                with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w') as f:
                    f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
        
            coef_pvalues = model.pvalues[1:] # 获取解释变量回归系数的p值
        
            if(np.all(coef_pvalues < significance_level)): # 如果模型直接全部参数显著，则放弃迭代直接出结果
                return stepwise_df

            else: # 如果存在不显著的变量，那么就随机剔除一个，再进行一次后退法。
                
                # 获取p值大于significance_level的变量索引
                insignificant_vars = np.where(coef_pvalues > significance_level)[0] 
                
                # 随机剔除不显著变量当中的随机一个变量
                random_index = random.choice(insignificant_vars) 
                
                # 从labels当中剔除这个变量的名称
                Labels.remove(Labels[random_index]) 
                
                if(len(Labels) == 0):
                    return stepwise_df # 此时说明怎么样都不显著，退无可退了
            
    else :  # 如果初始化不为空，则意味着此时已经完成了初始化变量的筛选
            # 此时说明存在模型使得模型不存在异方差和自相关现象，所以可以加入新的变量寻求更好的解释性，此时仍旧优先考虑模型无异方差和自相关
            # 进行前进法
            # 基本过程为： 随机从left_labels当中选取一个label加入init_labels并从left_labels当中剔除（如果left_labels为空，则结束循环并返回结果） -> 生成模型方程式 -> 回归并进行相关test -> 
            #                                           如果通过了异方差和自相关检验，则初步说明模型可用，记录；反之说明这个变量的加入会引起异方差和自相关，将该变量从init_labels当中剔除 -> 重复第一步
            # 注： 该方法可能忽略一些变量，为前进法，效果更好更为具体的模型可能仍需研究者个性化地进行寻找。

            while len(left_labels) > 0:
                random_label = random.choice(left_labels) # 从备选labels当中随机选取一个label
                init_labels = init_labels.append(random_label) # 将其加入init_labels当中
                left_labels.remove(random_label) # 将其从left_labels当中剔除
                
                formula = Formula_encoder([explained_vari] + init_labels)# 依照init_labels生成方程式
                model = smf.ols(formula , data = data).fit()
                Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model , labels = init_labels , higher_term = higher_term)
                
                if Bool_list[1] and Bool_list[2]: # 如果通过了检验，则init_labels当中加入这个变量，并且记录
                    stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                            spearman_labels , GQ_labels , 
                                                            np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                        ignore_index=True)
                    
                    if summary_output: # 如果需要输出则输出
                        with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w') as f:
                            f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
                            
                else: # 如果没有通过，则再次剔除这个变量，并且记录
                    init_labels.remove(random_label)
                    stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                            spearman_labels , GQ_labels , 
                                                            np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) ,
                                                        ignore_index=True)
                    if summary_output: # 如果需要输出则输出
                        with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w') as f:
                            f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
                            
            return stepwise_df # while循环后结束，并返回对应的结果



#### 岭回归

def Ridge_trace_analysis(data , explained_var = None , k = np.arange(0 , 100 , 1) , ax = None):
    """
    传入dataframe类型的data,默认第一列为被解释变量，如果需要指定被解释变量，输入对应的列名称即可，为字符串
    该函数主要进行岭回归,返回一个多维np矩阵,第一列为取的k的范围,默认为0-100,步长为1,剩下各列为回归系数在不同k取值下的岭回归值。
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
            ax.plot(k, coef, lw=1.5, label=data.columns[i]) 
        ax.set_xlabel('k')
        ax.set_ylabel('value')
        ax.set_title('岭迹分析图')
        ax.legend()
    return ret
    
if __name__=="__main__":
    df1 = pd.read_csv("data/select_data.csv",encoding = "utf-8")
    df2 = pd.read_csv('data/label_train.csv',index_col=0)
    index = df2.index
    pre_data = df1.loc[index , ].join(df2['MATH'])
    pre_data = pre_data.drop(pre_data.columns[0] , axis=1)
    pre_data = pre_data[['MATH']+list(pre_data.columns[0:-1].values)]
    pre_data = pre_data.head(600).reset_index(drop=True)
    stepwise_df = Stepwise_reg(pre_data)
    stepwise_df.to_csv('result_df.csv')
    
    
    
    # model = smf.ols(Formula_encoder(pre_data.columns) , data = pre_data).fit()
    # Bool_value , Bool_list , spearman_label , GQ_label = Model_exception_test(model.resid , pre_data , model)
    # print(Bool_value , Bool_list , spearman_label , GQ_label)
    
    
    
    
    # model1_CY = smf.ols('rent ~ area + room + subway' , data =Chaoyang).fit()
    # fitvalue1_CY = model1_CY.fittedvalues
    # res = model1_CY.resid
    
    # df1 , df2 , df3 , df4  , df5= res_test(res , fitvalue1_CY , Chaoyang)
    # print(df1 ,'\n', df2 , '\n' , df3 , '\n' , df4  , '\n' , df5)
    
    
    # fig , axes = res_plot(res , fitvalue1_CY)
    # plt.show()
    
    # print(Endogeneity_test(model1_CY))
    
    
    # fig , axes = plt.subplots(1 , 1 , figsize = (8 , 8) , dpi = 100)
    # print(Ridge_trace_analysis(Chaoyang ,k = np.arange(0 , 10000 , 10) ,  ax = axes))
    # plt.show()
    