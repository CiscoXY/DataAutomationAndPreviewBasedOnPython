import sys
sys.path.append('./') # 添加当前文件的父目录到系统路径
from datetime import datetime
import os


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.covariance import EllipticEnvelope

from Analyzation_relative import Descriptive_statistics as DS

#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)


# description:
# 该文件主要用于数据预处理和相关信息的图表展示。
# 包含模块：
# 1.数据的缺失值统计，甄别，和处理
# 具体为:  缺失值统计个数并出饼图和条形图(针对不同列分别进行统计和绘图)
#         缺失值甄别（即返回含有缺失值行组成的dataframe）
#         缺失值处理,包含删除、插值。其中插值有中位数插值/均值插值/前插/后插
#                           其中对于分类型数据,插值方法有: 前插/后插/以剩余分布的数值频率为概率进行多项分布的生成
#         函数目标为：输入数据，输出数据的缺失值统计图（如果需要输出的话），输出含有缺失值行组成的dataframe，输出自动处理后的dataframe(如果需要处理的话)
# 2.数据的异常值统计，甄别，和处理
# 具体为:  异常值的甄别（主要为一维方向的异常值甄别和多维方向的异常值甄别）
#         对于一维数值型数据,提供2种方案:分位数甄别/3 sigma 甄别 
#         对于一维二分型数据,具体解决方案见函数内具体描述
#         对于一维多分类型数据,暂无较为泛泛用的解决方法

#         对于多维数值型数据,提供两种甄别方案: 置信椭球法和LocalOutlierFactor(LOF)法，可以在参数中选择

#         对于甄别出的异常值,有2种处理方案可供选择(如果需要处理的话) : 删除/替换(对于一维数值型,选用中位数或者均值或者众数替换)
#                                                                    (对于一维二分型数据,以非异常值部分的对应频率为概率进行生成)
#                                                                    (对于多维数值型数据,以均值向量作为替换)
#         其中,在数据为2维数值型数据的时候,额外提供散点图绘制和置信椭圆(或者LOF边缘线)图进行参考
def Nan_CountAndPlot(data , plot = True , filepath = None):
    """
    生成数据框,存储各列的缺失值情况
    根据这个数据框绘制data当中各列Nan的情况,条形图(以及饼图) 如果需要plot的话
    
    return : 含有缺失值行的index组成的series , 统计后的dataframe(2列,第一列是完整数据的数量,第二列是缺失值数量)
    
    
    Args:
        data (dataframe): 传入的数据框
        plot (bool, optional): 是否绘图   Defaults to True
        filepath (str, optional): 图片和dataframe需要保存的位置. 如果为False,则不进行保存,仅返回数据框。如果为None,则按系统路径自动保存. Defaults to None.
                            options : None / False / (str , the path you want to save)
    """
    # 统计每列缺失值个数,并构建结果数据框
    nan_df = data.isnull().sum().reset_index()
    nan_df.columns = ['columns' , 'Nan_count']
    nan_df['Complete_count'] = len(data) - nan_df['Nan_count']  # 完成return的数据框
    if plot : # 如果需要绘图的话，进行下述内容
        col_names = data.columns.to_list() # 获取列名
        
        time = datetime.now()
        
        current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
        if filepath == None:
            dir_path = './dataprocess_result_output/Nan_condition'+current_time
        else:
            dir_path = filepath
        if not os.path.exists(dir_path):  # 如果不存在该文件夹，则创建该文件夹。
            os.makedirs(dir_path)
        
        figsize = (8 , 4) ; dpi = 100
        
        
        for index , col in enumerate(col_names):
            fig , axes = plt.subplots(1 , 2 , figsize = figsize , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化 
            count = nan_df.iloc[index , 1:]
            axes[0].pie(count , labels = ['Nan' , 'Complete'] , autopct = '%1.2f%%' , colors = ['red' , (84/255, 158/255, 227/255)])
            axes[0].axis('equal')
            axes[0].set_xlabel(col)
            axes[1].bar(count.index , count, color = (84/255, 158/255, 227/255) , width = 0.4)
            axes[1].set_xticks(count.index)
            for x,y in zip(count.index,count):
                axes[1].text(x,y+0.5,str(y),ha='center',va='bottom')
            axes[1].set_xlabel(col)

            if filepath != False:
                plt.savefig(dir_path + '/Nan_count of '+ col + '.png' , dpi = dpi)
    
    return data.index[data.isnull().any(axis=1)] , nan_df


def Nan_process(data , dataclass = None , plot = True , filepath = None , numeric = 'median' , subtype = 'frequency'):
    """
    函数目标为:输入数据  , 输出数据的缺失值统计图(如果需要输出的话) ; 输出含有缺失值行组成的dataframe , 输出自动处理后的dataframe(如果需要处理的话)
    
    
    Args:
        data (dataframe): 要进行缺失值处理的数据框
        dataclass (list, optional): data每列对应的数据类型,如果为None,则自动调用DS.DataLabeling. Defaults to None.
        plot (bool, optional): 是否进行绘图. Defaults to True.
        filepath (str, optional): 图片和dataframe需要保存的位置. 如果为False,则不进行保存,如果为None,则按系统路径自动保存. Defaults to None.
                            options : None / False / (str , the path you want to save)
        numeric(str , optional): 处理数值型数据选用的方法,可供选项有['median' , 'mean' , 'forward' , 'backward' , 'delete'] . Defaults to 'median'
        subtype(str , optional): 处理数值型数据选用的方法,可供选项有['frequency' , 'forward' , 'backward' , 'delete'] . Defaults to 'frequency'
    """
    if not numeric in ['median' , 'mean' , 'forward' , 'backward' , 'delete']:
        raise ValueError('Please input the correct type of numeric')
    if not subtype in ['frequency' , 'forward' , 'backward' , 'delete']:
        raise ValueError('Please input the correct type of subtype')
    
    nan_index , nan_df = Nan_CountAndPlot(data, plot = plot , filepath = filepath)
    
    if dataclass == None:
        dataclass = DS.DataLabeling(data)
    
    processed_df = data.copy()
    for index , value in enumerate(dataclass):
        col_nan_index = processed_df[processed_df.iloc[:,index].isnull()].index # 获取这列当中缺失值对应的索引
        if(value == 0):
            if numeric == 'median':
                processed_df.iloc[: , index].fillna(processed_df.iloc[: , index].median(), inplace = True)
                
            elif numeric == 'mean':
                processed_df.iloc[: , index].fillna(processed_df.iloc[: , index].mean(), inplace = True)
                
            elif numeric == 'forward':
                processed_df.iloc[: , index].fillna(method = 'ffill' , inplace = True)
                
            elif numeric == 'backward':
                processed_df.iloc[: , index].fillna(method = 'bfill' , inplace = True)
                
            elif numeric == 'delete':
                processed_df.drop(index = col_nan_index, axis = 0 , inplace = True)
        else : # 此时数据类型为1或者2，对应二分类型和多分类型
            if subtype == 'frequency':
                value_counts = processed_df.iloc[:, index].value_counts() # 统计各个类别出现的次数
                probs = value_counts / value_counts.sum() # 计算频率
                values = np.random.choice(value_counts.index, size = len(col_nan_index), p = probs)   # 以频率为概率进行多项抽样。
                processed_df.loc[col_nan_index, processed_df.columns[index]] = values # 完成缺失值替换
            elif subtype == 'forward':
                processed_df.iloc[: , index].fillna(method = 'ffill' , inplace = True)
                
            elif subtype == 'backward':
                processed_df.iloc[: , index].fillna(method = 'bfill' , inplace = True)
            
            elif subtype == 'delete': # 删除行
                processed_df.drop(index = col_nan_index, axis = 0 , inplace = True)
    
    
    return  data.loc[nan_index] , processed_df

def Outlier_TreatAndPlot(data , dataclass = None , plot = True , filepath = None , 
                            num_method = 'quantile' , num_treat = 'delete' , binary_boundary = 0.95 , binary_treat = 'delete' , 
                            multi_num_method = 'MCD' , contamination = 0.05 , multi_num_treat = 'delete'):
    """ 
    对于一维数值型数据,提供2种方案:分位数甄别/3 sigma 甄别 
    对于一维二分型数据, 如果本身数据有3个种类,但是前两个种类的占比超过 95% 则高度怀疑第三个种类是异常值
    对于一维多分类型数据,暂无较为泛泛用的解决方法,请手动识别

    对于多维数值型数据,提供两种甄别方案: 置信椭球法和LocalOutlierFactor(LOF)法，可以在参数中选择

    对于甄别出的异常值,有2种处理方案可供选择(如果需要处理的话) : 删除/替换  (对于一维数值型,选用中位数或者均值或者众数替换)
                                                                (对于一维二分型数据,以非异常值部分的对应频率为概率进行生成)
                                                                (对于多维数值型数据,以均值向量作为替换)
        其中,在数据为2维数值型数据的时候,额外提供散点图绘制和置信椭圆(或者LOF边缘线)图进行参考


    return : num_outlierindex(对于数值型数据识别为异常值的index) , type_outlierindex(对于二分类数据鉴定为异常值的index) , processed_df(处理完异常值后的数据)
    
    Args:
        data (dataframe): dataframe
        dataclass (list, optional): 这个dataframe对应的dataclass 如果为None则会自动识别(可能出错并且此时无法甄别二分类型的异常值). Defaults to None.
        plot (bool, optional): 是否画图. Defaults to True.
        filepath (str, optional): 要保存的文件路径. Defaults to None.
        num_method(str , optional): 对于一维数值型数据的甄别方法,默认为分位数法,即Q3+1.5IQR和Q1-1.5IQR外的数据认定为异常值 也可以使用Z分数法,即认为标准化后数据的Zscore得分的绝对值>3的为异常值 可选参数['quantile' , 'zscore'] . Default to 'quantile'
        num_treat(str , optional) : 对于异常值的处理 , 默认是删除,可选用均值或者中位数进行替换 可选参数['delete' , 'median' , 'mean'] . Default to 'delete'
        binary_boundary(str , optional) : 对于二分类型变量的临界值,如果占比最多的两个变量的占比比例超过这个临界值,则认定剩余的种类为异常值
        binary_treat(str , optional) : 对于二分类型变量的异常值的处理 , 可选参数['delete' , None] , None认定为不做处理。 Default to 'delete'
        multi_num_method(str , optional) : 对于多维数值型变量的甄别方法, 可选参数['MCD' , 'MD']. Default to 'MCD' , MCD为Minimum Covariance Determinant/ MD 则是常见的置信椭圆
        contamination(float , optional) : 多维数值型数据当中认为的异常值占比,默认为0.05
        multi_num_treat(str , optional) : 对于多维数值型变量异常值的处理方法, 可选参数['delete' , 'mean']. 'mean'为使用均指向量作替换 . Default to 'delete'
    """
    
    if isinstance(data , pd.Series):
        raise ValueError('Please trans this Series into Dataframe')
    elif isinstance(data , pd.DataFrame):
        col_names = data.columns.to_list()
        p = len(col_names)
        new_data , new_class = DS.data_sort(data , dataclass)
    else:
        raise ValueError('You must input Series or Dataframe')
    
    if dataclass == None:
        new_class = DS.DataLabeling(data)
        new_data , new_class = DS.data_sort(data , new_class)
    else:
        new_data , new_class = DS.data_sort(data , dataclass)
    
    time = datetime.now()
        
    current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
    if filepath == None:
        dir_path = './dataprocess_result_output/Outlier'+current_time
    else:
        dir_path = filepath
    if not os.path.exists(dir_path):  # 如果不存在该文件夹，则创建该文件夹。
        os.makedirs(dir_path)
    
    if 1 in new_class : # 如果类型中存在二分类
        numerical_class = new_class[:new_class.index(1)]
        type_class = new_class[new_class.index(1):]
    elif 2 in new_class : # 如果类型中不存在二分类，只存在多分类
        numerical_class = new_class[:new_class.index(2)]
        type_class = new_class[new_class.index(2):]
    else:
        numerical_class = new_class
        type_class = []
        type_outlierindex = []
    num_len = len(numerical_class) ; type_len = len(type_class) # 获取数值型数据的个数以及分类型数据的个数
    processed_df = new_data.copy()
    
    # 对于数值型数据进行甄别和处理
    if num_len == 0: pass
    elif num_len == 1: 
        if num_method == 'quantile':
            # 分位数甄别法
            IQR = processed_df.quantile(0.75) - processed_df.quantile(0.25) 
            temp = (processed_df < (processed_df.quantile(0.25) - 1.5 * IQR)) | (processed_df > (processed_df.quantile(0.75) + 1.5 * IQR))
            num_outlierindex = temp[temp.iloc[: , 0] == True].index
        elif num_method == 'zscore':
            # z得分检验法,标准化后值的绝对值如果超过3则认为是异常值
            mean, std = processed_df.mean(), processed_df.std()
            num_outlierindex = processed_df[np.abs(processed_df - mean) > 3*std].index 
        else:
            raise ValueError('You must choose a correct method')
        
        if num_treat == 'delete':
            processed_df.drop(index = num_outlierindex , axis = 0 , inplace = True)
        elif num_treat == 'median':
            processed_df[processed_df.index==num_outlierindex] = processed_df.median()
        elif num_treat == 'mean':
            processed_df[processed_df.index==num_outlierindex] = processed_df.mean()
        else:
            raise ValueError('You must choose a correct treat method')
    else : # 多维数值型数据
        classifiers = {
                            "Empirical Covariance": EllipticEnvelope(support_fraction=1.0, contamination=contamination),
                            "Robust Covariance(MCD)": EllipticEnvelope(support_fraction = 0.75 , 
                                contamination=contamination
                            )    # contamination : 数据的污染程度，也就是异常值的比例 , 此处取0.05   support_fraction : 可以理解为平滑程度,为1的话就不平滑，数值越小则越平滑，当为0.5时最鲁棒，但是高维度渐进效率太低,此处取0.8
                        }
        X = processed_df.to_numpy()
        # 如果绘图:
        if plot and num_len == 2:
            for (clf_name , clf) in classifiers.items():
                # 进行模型拟合
                clf.fit(X)
                
            # 获得最大最小值和极差 (绘图部分)
            max1  ,  min1= processed_df.iloc[: , 0].max() , processed_df.iloc[: , 0].min()
            max2 , min2 = processed_df.iloc[: , 1].max() , processed_df.iloc[: , 1].min()
            range1 , range2 = max1 - min1 , max2 - min2  

            legend = {} ; color = [(219/255, 151/255, 72/255) , (88/255, 219/255, 68/255)]
                
            xx1, yy1 = np.meshgrid(np.linspace(min1-0.1*range1, max1 + 0.1*range1, len(processed_df)), np.linspace(min2 - 0.1*range2, max2 + 0.1*range2, len(processed_df)))
            for i, (clf_name, clf) in enumerate(classifiers.items()):
                plt.figure(1)
                Z = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
                Z = Z.reshape(xx1.shape)
                legend[clf_name] = plt.contour(
                    xx1, yy1, Z, levels=[0], linewidths=2, colors= color[i]
                )
                
            legend1_values_list = list(legend.values())
            legend1_keys_list = list(legend.keys())
            
            if multi_num_method == 'MCD':
                outlier_score = classifiers['Robust Covariance(MCD)'].decision_function(X)
            elif multi_num_method == 'MD':
                outlier_score = classifiers['Empirical Covariance'].decision_function(X)
            else:
                raise ValueError('You must input correct multi_num_method')

            num_outlierindex = np.where(outlier_score == True)
            
        else:
            if multi_num_method == 'MCD':
                model = classifiers["Robust Covariance(MCD)"]
                model.fit(X)
                outlier_score = model.decision_function(X)
                
            elif multi_num_method == 'MD':
                model = classifiers['Empirical Covariance']
                model.fit(X)
                outlier_score = model.decision_function(X)
            else:
                raise ValueError('You must input correct multi_num_method')
            
            num_outlierindex = np.where(outlier_score == True)
        
        return 
            
            
    return num_outlierindex , type_outlierindex , processed_df



if __name__ == '__main__':
    df = pd.read_csv('data/wine/winequality-white-nan.csv' , index_col=0)
    df_withNan , df_pro = Nan_process(df , plot = False , numeric = 'delete')
