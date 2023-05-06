import sys
sys.path.append('./')
from datetime import datetime
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from multiprocessing import Pool # 多线程

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools   #* 用于进行解释变量名称的遍历。
import random

import statsmodels.api as sm
from  statsmodels.api import ProbPlot
from scipy.stats import spearmanr

from Analyzation_relative.Statistical_inference import Normality_test
from Analyzation_relative.Descriptive_statistics import DataLabeling , data_sort

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

def data_constructor(data , dataclass = None , target_type = 'linear' , degree = 2):
    """
    依照目标类型构建新数据框 , 可供选择的有
    线性模型 -> 原数据框不动
    带虚拟变量交互的线性模型 -> 使用数值型变量和二分类型变量构造交互项(不含有多分类变量的交互项,如果需要的话可以手动构造)
    多项式模型 -> 额外构造数值型变量的幂次项 , 默认为3次 
    带虚拟变量交互的多项式模型 -> 在构造完成多项式模型后,对每个数值型项额外进行虚拟变量交互项的引入.
    
    上述四个模型分别对应 : ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial']
    
    其中,当选定为linear时 , 并不会添加新的行 , 但是对于原始data中非数值型的分类型变量 , 会做分类型变量的数值化处理 (比方说一个变量有3个类 , A B C , 那么就会替换成 0 , 1 , 2)
    很明显 , 除了2分类型变量会被替换成0 , 1   多分类变量替换后会明显出现数值型的倾向, 所以会严重影响模型效果 , 故而!!!!
    
    # * 由于多分类型变量的特殊性 , 建议用户自行处理
    
    # * 注意 , 多项式方法会造成维数爆炸 , 带虚拟变量交互的多项式模型更会造成维数爆炸, 例如: 原始数据为2个数值型变量1个二分类型变量 , 那么此时的新数据阵就是2+1+2+(2+2)*1 = 9维
    # * 如果是更高维数的初始阵例如 : 3个数值型变量2个二分类型变量 , 那么新数据阵是 3+2+3+(3+3)*2 = 20维
    
    Args:
        data (dataframe): 需要构建的原始dataframe
        dataclass (list, optional): 输入dataframe对应每行的数据类型。如果为None则会自动识别. Defaults to None.
        target_type (str, optional): 目标的形式 可选系数 : ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial']. Defaults to linear.
        degree(int , optional): 幂次项的最高系数. Defaults to 2
    Returns:
        df_augmented: 添加交互项或者幂次项的dataframe
    """
    df_augmented = data.copy()
    
    # 获取排序后的dataframe和数据分类
    
    if dataclass == None:# 如果没有传入dataclass , 则进行自动判断
        new_class = DataLabeling(data) 
    else:
        if len(data.columns) != len(dataclass):
            raise ValueError('The length of columns and dataclass are not the same, please check')
        new_class = dataclass
    df_augmented , new_class = data_sort(df_augmented , dataclass = new_class) # 进行排序
    
    # 获取各种类型变量的列名
    col_names = df_augmented.columns.to_list()
    if 1 in new_class: # 如果有二分类变量
        num_columns = col_names[:new_class.index(1)]
        
        if 2 in new_class: # 如果还有多分类型变量
            binary_columns = col_names[new_class.index(1) : new_class.index(2)]
            multitype_columns = col_names[new_class.index(2) : ]
            
        else:
            binary_columns = col_names[new_class.index(1) : ]
            multitype_columns = []
        
    elif 2 in new_class: # 如果没有二分类型变量，但是有多分类型变量
        num_columns = col_names[:new_class.index(2)]
        binary_columns = []
        multitype_columns = col_names[new_class.index(2) : ]
    else: #如果全是数值型变量
        num_columns = col_names
        binary_columns = []
        multitype_columns = []
    
    
    # 对分类型变量做映射
    
    for col in (binary_columns + multitype_columns):
        if isinstance(df_augmented[col].dtype, pd.CategoricalDtype):# 如果本身已经是category了，那么不做处理
            pass
        else :
            df_augmented[col] = df_augmented[col].astype('category') # 转换为category
            df_augmented[col] = df_augmented[col].cat.codes # 变换标签
    
    # 如果是线性模型
    if target_type == 'linear':
        # 此时原封不动返还数据即可
        return df_augmented

    # 如果是带虚拟变量交互的模型
    elif target_type == 'dummy and linear':
        # 先查验是否有2分类型变量
        if len(binary_columns) == 0:
            raise ValueError('There isn\'t any binary variable , please check')
        
        # 如果有2分类型变量
        else:
            # 如果没有数值型变量M , 自然不含交互项
            if len(num_columns) == 0:
                raise ValueError('There isn\'t any numeric variable , please check')
            
            # 如果含有数值项,则构造数值项和二分类变量的交互项
            else:
                for bina_col in binary_columns:
                    # 每个二分类变量都与剩下的数值项交互
                    for num_col in num_columns:
                        add_col_name = f'{num_col}*{bina_col}'
                        add_col = df_augmented[num_col] * df_augmented[bina_col]
                        add_col.name = add_col_name
                        df_augmented = pd.concat([df_augmented, add_col], axis=1)
        
        # 返回添加了虚拟变量交互项的data
        return df_augmented
        
    elif target_type == 'polynomial' or target_type == 'dummy and polynomial':
        # 先检查是否有数值型变量
        if len(num_columns) == 0:
            raise ValueError('There isn\'t any numeric variable , please check')
        
        # 如果已经存在数值型变量了
        else:
            # 检查输入参数是否合规
            if degree<=1:
                raise ValueError('degree must >= 2')
            
            # 检查是否有数值型变量
            if len(num_columns) == 0:
                raise ValueError('There isn\'t any numeric variable , please check') 
            
            else:
                add_col_list = []
                
                for col in num_columns: # 对每个数值型变量都要进行augment
                    for i in range(2 , degree+1):
                        add_col_name = f'{col}^{i}'
                        add_col_list.append(add_col_name) # 增加列名
                        add_col = df_augmented[col] ** i
                        add_col.name = add_col_name
                        df_augmented = pd.concat([df_augmented, add_col], axis=1)
                
                num_columns.extend(add_col_list) # 将增加的列名加到数值型当中
        
        # 如果是多项式则不需要操作
        if target_type == 'polynomial':
            pass
        
        # 如果是带虚拟变量交互项的多项式
        else:
            # 如果没有二分类型变量
            if len(binary_columns) == 0:
                raise ValueError('There isn\'t any binary variable , please check')
            
            # 如果有二分类型变量
            else:
                for bina_col in binary_columns:
                    # 每个二分类变量都与剩下的数值项交互
                    for num_col in num_columns:
                        add_col_name = f'{num_col}*{bina_col}'
                        add_col = df_augmented[num_col] * df_augmented[bina_col]
                        add_col.name = add_col_name
                        df_augmented = pd.concat([df_augmented, add_col], axis=1)
        
        # 返回添加了高次项和虚拟变量交互项的data
        return df_augmented
    else : 
        raise ValueError('You must input correct target_type')


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
        # 如果该列的唯一值只有两个，并且其中一个，则无法进行white检验，因为会出现严格多重共线性
        if len(unique_values) == 2 and any(x == 0 for x in unique_values):
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
def Multicollinearity_test(data , VIF = True):
    """
    用于检验多重共线性,data应当全是解释变量组成的dataframe或者np.array
    返回的dataframe由两部分组成
    方差扩大因子法和特征值（条件指数法）`
    
    VIF (bool , optional) : 是否使用方差扩大因子法求解 , 如果为False则只用条件指数
    """
    Mul_df = pd.DataFrame()
    
    if isinstance(data, pd.DataFrame):
        Columns = data.columns
        X = data.values
    else: 
        X = data
        Columns = [f'x{i+1}' for i in range(X.shape[1])]
    
    # 条件指数
    XTX = X.T @ X 
    eigenvalues = np.linalg.eigvals(XTX)
    max_eig = np.max(eigenvalues)  
    min_eig = np.min(eigenvalues)
    con_index = np.sqrt(max_eig/min_eig)
    # 方差扩大因子法
    if VIF:
        vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        index = [f'{col}_VIF' for col in Columns] + ['条件指数']
        Mul_df = pd.DataFrame({'value': vif + [con_index]}, index=index)
    else:
        Mul_df = pd.DataFrame({'value' : [con_index]} , index = ['条件指数'])
    
    
    
    
    return Mul_df

#### 逐步回归(考虑到各种异常情况的)
    # 注： 以下的逐步回归会优先考虑模型的假定问题，正态性，异方差性，自相关性以及内生性假定问题，随后才是参数的显著性问题。
def Stepwise_reg(data ,filepath = None , summary_output = True , result_output = True ,  significance_level = 0.05 , higher_term = True):
    """
    该函数主要对数据data进行逐步回归选择,优先考虑模型的各种异常情况，
    
    data : 数据阵,默认第一列是被解释变量
    filepath(str , optional) : 保存文件的路径(最后会加上时间) . Default to None
    summary_output : 是否将筛选的模型的summary以txt文件格式输出,默认为True,输出文件的目录为调用该模块的main.py函数的同一目录下的'result_output'文件夹内 . Default = True
    result_output : 是否将每步筛选的到的dataframe输出. Default = True
    significance_level : 判断是否通过检验的显著性水平
    higher_term : 数据是否含有高维项,如果有则在检验时会忽略white检验和内生性检验
    
    return :dataframe   第1列是该formula是否通过了异常情况检验
                        第2,3,4,5列分别是正态性,异方差,自相关,内生性的检验通过与否的bool值
                        第6列存储着spearman秩相关检验下引起异方差的自变量名称,一个list,如果没有则为空
                        第7列存储着Goldfeld-Quandt检验下引起异方差的自变量名称,一个list,如果没有则为空
                        第8列存储着该formula对应的各种参数的t检验p值
                        第9列是该formula的调整的R方
                        第10列是F统计量的p值
                        第11列是formula,也就是对应的方程表达式,一串string
    """
    # 获得系统时间
    time = datetime.now()
    
    current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
    
    if filepath == None:
        dir_path = './reg_result_output/'+current_time
    else:
        dir_path = filepath + '/' + current_time

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
    for i in range(int(len(Labels) * 0.75) + 1): # 进行最多变量数*0.75的选取，如果都没有选择到不存在异方差和自相关的初始化label
        init_labels , left_labels = random_subset(Labels , k = 2)
        formula = Formula_encoder([explained_vari] + init_labels)
        print('init_formula is '+ formula)
        model = sm.OLS(data[explained_vari] , sm.add_constant(data[init_labels])).fit()
        Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model ,labels = init_labels , higher_term = higher_term)
        if Bool_list[1] and Bool_list[2]:
            # 如果不存在 异方差和自相关 则初始化完成，像最终数据框中添加数据并跳出循环
            stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                        spearman_labels , GQ_labels , 
                                                        np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                ignore_index=True)
            if summary_output: # 如果需要输出则输出
                with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w', encoding='utf-8') as f:
                    f.write(model.summary().as_text() , )   # 向目标文件夹内的文件输出对应的summary并形成单独文件
            
            break
        else:
            init_labels = []
    if(len(init_labels) == 0) : #如果初始化labels空，则说明数据和模型本身存在极大异常，需要重新处理数据或者换模型。此时不优先考虑模型的基本假定相关检验（异方差，自相关，内生性等）
                                #反而根据参数的显著性，从全模型开始进行后退法筛选变量。
                                #基本过程为，全模型 -> 参数显著性检验 -> 随机剔除一个不显著的参数（如果没有不显著的参数了就停止迭代）-> 回归后做基本假定相关test并记录 -> 下一次迭代。
        while True: #开始后退法的循环
        
            formula = Formula_encoder([explained_vari] + Labels) # 依照Labels模型的方程式
            model = sm.OLS(data[explained_vari] , sm.add_constant(data[Labels])).fit()
            Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model ,labels = Labels , higher_term = higher_term)
        
            stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                        spearman_labels , GQ_labels , 
                                                        np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                ignore_index=True)
            if summary_output: # 如果需要输出则输出
                with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w', encoding='utf-8') as f:
                    f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
        
            coef_pvalues = model.pvalues[1:] # 获取解释变量回归系数的p值
        
            if(np.all(coef_pvalues < significance_level)): # 如果模型直接全部参数显著，则放弃迭代直接出结果
                if result_output: stepwise_df.to_excel(dir_path + '/result.xlsx' , encoding='utf-8')
                return stepwise_df

            else: # 如果存在不显著的变量，那么就随机剔除一个，再进行一次后退法。
                
                # 获取p值大于significance_level的变量索引
                insignificant_vars = np.where(coef_pvalues > significance_level)[0] 
                
                # 随机剔除不显著变量当中的随机一个变量
                random_index = random.choice(insignificant_vars) 
                
                # 从labels当中剔除这个变量的名称
                Labels.remove(Labels[random_index]) 
                
                if(len(Labels) == 0):
                    if result_output: stepwise_df.to_excel(dir_path + '/result.xlsx' , encoding='utf-8')
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
                model = sm.OLS(data[explained_vari] , sm.add_constant(data[init_labels])).fit()
                Bool_value , Bool_list , spearman_labels , GQ_labels = Model_exception_test(model.resid , data , model , labels = init_labels , higher_term = higher_term)
                
                if Bool_list[1] and Bool_list[2]: # 如果通过了检验，则init_labels当中加入这个变量，并且记录
                    stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                            spearman_labels , GQ_labels , 
                                                            np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) , 
                                                        ignore_index=True)
                    
                    if summary_output: # 如果需要输出则输出
                        with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w' , encoding='utf-8') as f:
                            f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
                            
                else: # 如果没有通过，则再次剔除这个变量，并且记录
                    init_labels.remove(random_label)
                    stepwise_df = stepwise_df.append(pd.DataFrame([[Bool_value , Bool_list[0] , Bool_list[1] , Bool_list[2] , Bool_list[3] , 
                                                            spearman_labels , GQ_labels , 
                                                            np.round(model.pvalues.to_list() , 4) , round(model.rsquared_adj , 4) , round(model.f_pvalue , 5) , formula]] , columns = stepwise_columns) ,
                                                        ignore_index=True)
                    if summary_output: # 如果需要输出则输出
                        with open(dir_path+'/formula'+str(stepwise_df.shape[0])+'.txt' , 'w' , encoding='utf-8') as f:
                            f.write(model.summary().as_text())   # 向目标文件夹内的文件输出对应的summary并形成单独文件
                            
            if result_output: stepwise_df.to_excel(dir_path + '/result.xlsx' , encoding='utf-8')
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
    

#   # TODO 多线程   明天写  claude牛逼！
# 多线程模块描述：
# 输入原始data ，输入想要自动判断的模式，传递参数为一个list，list当中存储着想要全自动回归的模型类别。 随后再特定文件路径内输出全自动回归的判断过程和结果，所用函数为 Stepwise_reg

# 可供选择的模型类别为 : ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial'] 分别对应 线性模型 ， 带虚拟变量交互项的线性模型 ， 多项式 ， 带虚拟变量交互的多项式




def Automatic_reg(data , dataclass = None , target_col = None , mode = None , filepath = None):
    """ 多线程模块
    针对data进行全自动回归 . 并在指定文件夹内输出结果 . 文件路径如果不指定则默认为 './Automatic_reg.result'
    
    当mode不为None或长度不为1时 , 会调用多线程

    Args:
        data (dataframe): 输入的原始数据(默认为不存在缺失值的数据 , 如果存在则会返回valueerror)
        
        dataclass (list, optional): 存储着data每列的数据类型 , 0为数值型 , 1为二分类型 , 2为多分类型; 如果为None , 则会自行判断. Defaults to None.
        
        target_col (str, optional): 被解释变量的列名 ; 如果为None则默认第一列为被解释变量. Defaults to None.
        
        mode (list, optional): 存储着想要回归的目标模型的类别 , 应当为list , 可选参数 : ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial'] 
        当为None时会默认全部并且自动进行删减判断.  当参数长度不为1时 , 会调用多线程进行处理. Defaults to None.
        
        filepath (str, optional): 想要保存的文件的路径 , 结果会在这个文件路径内部开辟子文件夹 , 名称为所探究的模型类型,  并依照执行时间进行自动判断. Defaults to None.
        
    """
    # 对data进行格式和维数审查 , 如果不符合则valueerror
    if not isinstance(data , pd.DataFrame):
        raise ValueError('data must be a dataframe')
    else:
        if data.shape[1] < 2:
            raise ValueError('data\'s dimensions must >= 2')
        elif data.shape[0] <= data.shape[1]:
            raise ValueError('data has dimension explosion , please check')
        # 此时data是数据框且至少2列且不存在维数爆炸
    
    # 获取数据类型
    if dataclass == None:# 如果没有传入dataclass , 则进行自动判断
        new_class = DataLabeling(data) 
    else:
        if not isinstance(dataclass , list):
            raise ValueError('The dataclass must be a list')
        if len(data.columns) != len(dataclass):
            raise ValueError('The length of columns and dataclass are not the same, please check')
        if not all(elem in [0, 1, 2] for elem in dataclass):
            raise ValueError('The dataclass must only include 0 , 1 , 2 , please check')
        new_class = dataclass
    
    
    # 调整数据顺序，使得被解释变量变为第一列
    col_names = data.columns.to_list() # 获取列名转换为list
    if target_col == None:
        df = data.copy()
        pass # 如果默认，则认为数据第一列是
    else:
        if target_col in col_names:
            new_class.insert(0, new_class.pop(col_names.index(target_col))) # 将target_col对应的dataclass的元素放到第一位
            col_names.remove(target_col)
            col_names.insert(0 , target_col)
            df = data[col_names] # 完成了被解释变量列换到第一列的任务
        else:
            raise ValueError(f'{target_col} is not in data , please check')
    
    #*  至此已经完成了数据、数据类型的诊断和整理
    
    data_y , data_X = df.iloc[: , 0] , df.iloc[: , 1:] # 被解释变量和解释变量阵的分割，方便之后做变换
    class_y , class_X = [new_class[0]] , new_class[1:] # 获得被解释变量和解释变量的类别 同时对被解释变量的类别进行排序 因为之后要的是顺序
    
    # 判断mode格式
    if mode == None :
        mode = ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial']
    else:
        if not isinstance(mode , list):
            raise ValueError('The mode must be a list')
        if len(mode) == 0:
            raise ValueError('mode can\'t be Null list')
        if not all(elem in ['linear' , 'dummy and linear' , 'polynomial' , 'dummy and polynomial'] for elem in mode):
            raise ValueError('The dataclass must only include \'linear\',\'dummy and linear\',\'polynomial\',\'dummy and polynomial\', please check')
        mode = list(set(mode)) # 只去唯一值，防止长度过多浪费时间
    
    # 判断mode当中的模式是否可用，不可用则抛出
    X_list = [] ; mode_copy = list(mode) # 拷贝一份
    for mode_type in mode:
        if mode_type == 'linear':
            X_list.append(data_constructor(data_X , dataclass = class_X)) # 如果为线性，则直接将结构化的解释变量加入解释变量list
        elif mode_type == 'dummy and linear' or mode_type =='dummy and polynomial':
            if not (1 in class_X):
                warnings.warn(f'There is no binary variable in X , ignored this mode : {mode_type}')
                mode_copy.remove(mode_type)
            elif not (0 in class_X):
                warnings.warn(f'There is no numeric variable in X , ignored this mode : {mode_type}')
                mode_copy.remove(mode_type)
            else:
                X_list.append(data_constructor(data_X , dataclass = class_X , target_type=mode_type))
        
        elif mode_type == 'polynomial':
            if not (0 in class_X):
                warnings.warn(f'There is no numeric variable in X , ignored this mode : {mode_type}')
                mode_copy.remove(mode_type)
            else:
                X_list.append(data_constructor(data_X , dataclass = class_X , target_type=mode_type))


    class_X = sorted(class_X) # 升序排序一下，这样才能一一对应X_list当中的头几个原始数据
    mode = mode_copy
    
    if len(mode) == 0:
        raise ValueError('Function can\'t word under the parameter of \'mode\' you give. \n Please check it again')
    
    # 创建各种文件路径
    
    if filepath == None:
        dir_path = './Automatic_reg.result'
    else:
        dir_path = filepath + '/Automatic_reg.result'

    # 根据mode创建对应的文件路径
    dir_list = []
    for mode_type in mode:
        sub_dir = dir_path + '/' + mode_type # 创建名为 Automatic_reg.result current_time/mode_type 的子文件夹
        dir_list.append(sub_dir)             # 此处完成了添加对应的文件路径到文件路径列表   以便后续的多线程进行.
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    
    # 根据mode设置各种stepwise_reg的关键字参数 , 并构建args_list
    args_list = []
    for index , mode_type in enumerate(mode):
        args = [pd.concat([data_y, X_list[index]], axis=1) , dir_list[index] , True ,True ,  0.05]
        if mode_type == 'linear' :
            if 1 in class_X:
                indices = [i for i, x in enumerate(class_X) if x == 1]  # 如果线性模型的解释变量当中存在二分变量，那么就针对这些二分变量讨论
                for i in indices:
                    unique_values = X_list[index].iloc[: , i].unique()  # 获得这个二分变量列的唯一值  之所以为2+1
                    # 如果该列的唯一值只有两个，并且其中一个还是0，则无法进行white和RESET检验，因为引入高阶项之后会出现严格多重共线性
                    if len(unique_values) == 2 and any(x == 0 for x in unique_values):
                        
                        args.append(True)  # 此时设置 higher_term = True 即模型中包含高阶参数，实际上不包含，只是为了不去做相关检验罢了
                        break
                
                if args[-1] == 0.05 : args.append(False)  # 如果args这个参数列表只有四个参数，说明前面的检验都通过了，可以引入高阶项进行检验
            else:
                args.append(False)# 如果为线性模型，则不包含高阶项，所以需要进行white和内生性检验

        else : 
            # 其他情况的模型，即'dummy and linear' , 'polynomial' , 'dummy and polynomial',这些情况下都包含了高阶项，所以higher_term = True
            args.append(True)
        args_list.append(args)
    
    # 完成了全部构建 , 开始主程序部分
    stepwise_resultdf_list = []
    if len(args_list) == 1: # 如果参数list只有一个，说明只需要进行一次Stepwise_reg , 故而此时不需要多线程
        stepwise_resultdf = Stepwise_reg(args_list[0][0] , filepath = args_list[0][1] , summary_output = args_list[0][2] , result_output = args_list[0][3] ,  significance_level = args_list[0][4] , higher_term = args_list[0][5])
        stepwise_resultdf_list = [stepwise_resultdf]
    
    
    # 调用多线程
    else:
        with Pool(len(args_list)) as p:  # 使用 4 个进程
            results = p.starmap(Stepwise_reg, args_list)
        stepwise_resultdf_list = results # 因为调用了多线程，所以此时result是无序的
    
    
    return mode , X_list , args_list , stepwise_resultdf_list
    



    
    
    
if __name__=="__main__":
    new_df = pd.read_csv('./data/test/display.csv' , index_col = 0)
    mode , Xlist , args_list , result_list = Automatic_reg(new_df , dataclass = None ,target_col = None ,  mode = None, filepath = './test/construct_data')
    
    
    # df = pd.read_excel("data/merge_shop_coupon_nm.xls" , index_col = 0)
    # columns = ['关键词', '城市', '评分', '评价数', '人均' , '团购价', '购买人数']
    # data = df[columns]
    
    # mode , Xlist , args_list , result_list = Reg.Automatic_reg(df[['评价数','城市' , '评分', '人均']] , dataclass = None , mode = ['linear'] , filepath = './test')
    
    # * 用北京房价数据进行多线程测试
    # * dataclass 为 [0, 2, 1, 1, 0, 1, 2, 1, 2, 1]
    # df = pd.read_csv("data/test_data.csv",encoding = "utf-8")  
    # mode , Xlist , args_list , result_list = Automatic_reg(df , dataclass = [0, 2, 1, 1, 0, 1, 2, 1, 2, 1] ,target_col = 'area' ,  mode = None , filepath = './test/construct_data')
    
    
    # df1 = pd.read_csv("data/select_data.csv",encoding = "utf-8")
    # df2 = pd.read_csv('data/label_train.csv',index_col=0)
    # index = df2.index
    # pre_data = df1.loc[index , ].join(df2['MATH'])
    # pre_data = pre_data.drop(pre_data.columns[0] , axis=1)
    # pre_data = pre_data[['MATH']+list(pre_data.columns[0:-1].values)]
    # pre_data = pre_data.head(600).reset_index(drop=True)
    # stepwise_df = Stepwise_reg(pre_data)
    # stepwise_df.to_csv('result_df.csv')
    
    
    
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
    