import sys
sys.path.append('./') # 添加当前文件的父目录到系统路径
from datetime import datetime
import os

from operator import eq

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from statsmodels.graphics import mosaicplot

from scipy.stats import chi2 , norm

from Analyzation_relative import Multivariate_statistical as MS
#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)
# 通用变量
glob_color = (84/255, 158/255, 227/255) # 各种绘图的填充颜色，浅蓝色



#### 小工具
def foo(data):
    """
    一个用于判断传入数据是dataframe类型还是array类型
    """
    if isinstance(data, pd.DataFrame):
        return 1
    elif isinstance(data, np.ndarray):
        return 0



# 给数据贴标签
def DataLabeling(data):  
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

# 根据上面函数标签后的标签，升序排序原始data

def data_sort(data , dataclass):
    """
    根据dataclass进行降序排列data

    Args:
        data (dataframe): 原始数据
        dataclass (list): 按照数据各列的属性生成的list,对应: 数值型 0 ; 二分类型 1 ; 多分类型 2
    
    return 整理后的新data , 整理后的新data对应的dataclass
    """
    columns = data.columns # 获取列名
    sorted_index = list(np.argsort(dataclass)) # 按升序排列dataclass，并获得排序后对应的原索引
    order_columns = [columns[i] for i in sorted_index]
    order_dataclass = [dataclass[i] for i in sorted_index]
    
    return data[order_columns] , order_dataclass
    





def chi2_QQ(X , axes):
    '''
    绘制卡方QQ图(对应维数)
    传进来一个n , p维矩阵 , (不能是数据框)
    和要绘制的axes(plt的subplot)
    '''
    if isinstance(X, pd.DataFrame):
        Columns = X.columns
        temp = X.values
    elif isinstance(X, np.ndarray):
        temp = X
    else : 
        raise ValueError('X must be Dataframe of ndarray \n 请输入一个Dataframe或者ndarray')
    if(X.ndim == 1):
        print('至少为2维才能绘制卡方图')
        exit(-1)
    n , p = X.shape
    chi2_plist = chi2.ppf((np.array(range(1,len(temp)+1))-0.5)/len(temp) , p)
    d = MS.Mahalanobis_Distance(temp)
    inf = np.min([d,chi2_plist]) ; sup = np.max([d,chi2_plist])
    x = np.arange(inf , sup + 0.1 , 0.1)
    axes.plot(x , x , color = 'green') #* 绘制y=x标准线
    axes.scatter(chi2_plist , np.sort(d) , s = 9 , alpha = 0.6)


##### 画图

### 复合特殊图

## 嵌套饼图

def Nesting_pie(data , ax = None):
    """
    绘制一个由二维多分类型变量组成的嵌套饼图
    具体形为内环和外环的环形图。  注意,传入的data的第一列默认为内环,如果需要更改内外环对应的变量,只需要更改data的第一二列的顺序即可
    
    data : 数据,二维的dataframe,原始数据即可
    ax : 需要绘制的子图对象
    """
    if(ax != None):
        column_names = data.columns
        col1_counts = data[column_names[0]].value_counts().sort_values(ascending=False) # 获取第一列的各个种类的降序排列
        col2_list = [] # 获取第二列的各个种类的降序排列组成的list，index顺序为对应第一列的种类
        for col1_type in col1_counts.index:
            df_col1_type = data.loc[data.iloc[: , 0] == col1_type , column_names[1]]
            col2_counts = df_col1_type.value_counts().sort_values(ascending = False) # 降序排列
            
            # 添加至col2_list
            col2_list.append(col2_counts)
        
        # 获取color
        colors_1 = sns.husl_palette(len(col1_counts))
        
        label_inner = list(col1_counts.index)
        label_outer = []
        outer_proportion = []
        outer_colors = []
        for series in  col2_list:
            label_outer.extend(list(series.index)) # 加入外环的label
            outer_proportion.extend(series.to_list()) # 加入外环的比例
            outer_colors.extend(sns.husl_palette(len(series))) # 加入外环的颜色
        
        ax.pie(col1_counts , labels = label_inner , autopct = '%1.1f%%' , colors = colors_1 , radius = 0.5 , labeldistance = 0.8 , 
                            wedgeprops=dict(width=0.5, edgecolor='w') , textprops={'fontsize': 7}) # 内环饼图
        ax.pie(outer_proportion , labels = label_outer , autopct = '%1.2f%%' , colors = outer_colors , radius = 1 , labeldistance = 1 , pctdistance=0.85 , 
                            wedgeprops=dict(width=0.5, edgecolor='w') , textprops={'fontsize': 7}) # 外环饼图
        ax.axis('equal')
        ax.set_title(column_names[0] + ' with ' + column_names[1])

def Mosaic_plt(data , ax = None):
    """
    绘制一个由2个分类型变量组成的马赛克图
    具体为横轴为第一列数据对应的变量种类,纵轴为第二列数据对应的变量种类
    """
    if( ax != None):
        columns = data.columns
        mosaicplot.mosaic(data , ax = ax , index = columns.to_list() , gap = 0.02 , labelizer=lambda k : k[1])
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_title('Mosaic '+columns[0] + '&' + columns[1])

### 一维数据图


## 数值型

def Numerical_autoplt(data , ax1 = None, ax2 = None):
    """
    绘制一个一维数值型数据的描述性统计图，其中
    
    data : 数据,最好为pd.Series
    ax1 : 绘制带核密度估计曲线的直方图的plt.subplot的子图 , 该图还包含: 对应正态分布的曲线
    ax2 : 绘制小提琴图的plt.subplot的子图对象
    """
    #绘制直方图与核密度估计曲线
    if (ax1 != None):
        mu , std , inf , sup = data.mean() , data.std() , data.min() , data.max()
        x = np.arange(inf , sup , 1/len(data))
        y = norm.pdf(x ,  mu , std) 
        sns.histplot(data ,color = glob_color ,  stat = 'density' ,alpha = 0.7 ,  ax = ax1)
        # 生成标准的正态曲线
        ax1.plot(x , y , "k--", lw=1.5 , label = 'Norm contrast')

        ax1.grid()
        ax1.legend()
        sns.kdeplot(data, color="r", lw=1 ,alpha = 0.7 ,  ax = ax1 , label = 'Kde line')
    
    # 绘制箱线图
    if(ax2 != None):
        sns.boxplot(x = data , ax = ax2 , color = glob_color)
        ax2.grid()
    

## 二分类型

def Binary_autoplt(data , ax1 = None , ax2 = None):
    """
    绘制一个一维二分类型数据的描述性统计图，其中
    
    data : 数据,最好为pd.Series
    ax1 : 绘制饼图
    ax2 : 绘制条形图
    """
    # 获取频数
    count = data.value_counts()
    name = count.name
    index = count.index
    
    # 获取color
    colors = sns.color_palette('pastel')[0:len(count)]
    
    if(ax1 != None):
    
        ax1.pie(count , labels = index , autopct = '%1.2f%%' , colors = colors , pctdistance=0.8)
        ax1.axis('equal')
        ax1.set_xlabel(name)
    
    if(ax2 != None):
        ax2.bar(index , count, color = glob_color , width = 0.4)
        ax2.set_xticks(index)
        for x,y in zip(index,count):
            ax2.text(x,y+0.5,str(y),ha='center',va='bottom')
        ax2.set_xlabel(name)

## 多分类型

def Multitpye_autoplt(data , ax1 = None , ax2 = None):
    """
    传入一个多分类型数据，绘制相关图形
    Args:
        data : 数据,最好为pd.Series
        ax1 : 绘制环形图
        ax2 : 绘制带累积分布曲线的帕累托图
    """
    # 获取频数,列名,标签等信息
    count = data.value_counts().sort_values(ascending = False)
    name = count.name
    index = count.index.to_list()
    index = [str(i) for i in index]
    cumulative_p = count.cumsum()/count.sum() * count.iat[0]
    # 获取color
    colors = sns.husl_palette(len(count))
    # 绘制饼图
    if(ax1 != None):
        ax1.pie(count , labels = index , autopct = '%1.2f%%' , colors = colors)
        ax1.axis('equal')
        ax1.set_xlabel(name)
    # 绘制帕累托图
    if(ax2 != None):
        ax2.bar(index , count , color = glob_color)
        ax2.set_xticks(index)
        ax2.plot(index , cumulative_p , color = 'g' , lw = 1.5,  linestyle = '--')
        for i , x in enumerate(index):
            ax2.text( i , count.iat[i] + 0.5 , str(count.iat[i]) , ha='center',va='bottom' , fontsize = 10)
            ax2.text(i , cumulative_p.iat[i] + 0.5 , str(round(cumulative_p.iat[i]/count.iat[0]*100 , 2)) + '%', ha='center',va='bottom' , fontsize = 9)
        ax2.set_xlabel(name)
        ax2.set_xticklabels(index , rotation = 90 , fontsize = 8)
    
### 二维数据

## 绘制一个二维数据的数据图

def Two_dim_autoplt(data , dataclass , filepath = None , save = True , show = False , figsize = None , dpi = None):
    """
    对一个二维数据进行描述性统计图绘制,并将结果结构化输出为图片保存到既定文件路径。
    Args:
        data : dataframe 应为2 维
        dataclass : list , 存储着对应位置的数据的类别, 具体为：数值型 0 ; 二分类型 1 ; 多分类型 2
        filepath : 想要保存到的文件夹路径 Defaults to None.
        save : 是否保存为.png , 默认为True
        show : 是否显示,默认为False 因为会阻塞进程
    """
    # 建立文件路径
    time = datetime.now()
    
    current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
    if filepath == None:
        dir_path = './descriptive_result_output/'+current_time
    else:
        dir_path = filepath
    if not os.path.exists(dir_path):  # 如果不存在该文件夹，则创建该文件夹。
        os.makedirs(dir_path)
    
    if dpi == None : dpi = 100


    new_data , new_class = data_sort(data , dataclass) # 对原data依照dataclass排序并获得排序后的data和class
    
    # 两个数值型变量
    if(eq(new_class ,[0 , 0])): 
        if figsize == None : figsize = (8 , 12)
        
        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Numerical_autoplt(new_data.iloc[: ,0] , ax1 = axes[0][0] , ax2 = axes[0][1]) # 数值型的两个图
        Numerical_autoplt(new_data.iloc[: , 1] , ax1 = axes[1][0] , ax2 = axes[1][1]) # 数值型的两个图
        
        sns.regplot(x = new_data.columns[0] , y=new_data.columns[1] , data = new_data , ax = axes[2][0] , scatter_kws={'s': 10, 'alpha': 0.5}) # 带回归线的散点图
        axes[2][0].grid()
        sns.kdeplot(x = new_data.iloc[: , 0], y = new_data.iloc[: , 1], shade=True , ax = axes[2][1]) # 二维核密度图
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()

    # 一个数值型一个二分类型
    elif(eq(new_class , [0 , 1])):
        if figsize == None : figsize = (8 , 12)

        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi
                                = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Numerical_autoplt(new_data.iloc[: , 0] , ax1 = axes[0][0] , ax2 = axes[0][1]) # 数值型的两个图
        Binary_autoplt(new_data.iloc[: , 1] , ax1 = axes[1][0] , ax2 = axes[1][1]) # 二分型的两个图
        
        Binary_value = new_data.iloc[: , 1].unique() # 获取这个二分类变量的两个值
        df_cat0 = new_data[new_data.iloc[: , 1] == Binary_value[0]] # 其中一个值的子集
        df_cat1 = new_data[new_data.iloc[: , 1] == Binary_value[1]] # 另一个值的子集
        
        Numerical_autoplt(df_cat0.iloc[: ,0] , ax1 = axes[2][0]) ; axes[2][0].set_title('For ' + new_data.columns[1] + ' = ' + str(Binary_value[0]))
        Numerical_autoplt(df_cat1.iloc[: ,0] , ax1 = axes[2][1]) ; axes[2][1].set_title('For ' + new_data.columns[1] + ' = ' + str(Binary_value[1]))
        
        plt.tight_layout() # 自动调整子图间距
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 绘制统一大图下，不同二分的各自的密度曲线
        fig , axes = plt.subplots(figsize = (6,6) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        sns.kdeplot(df_cat0.iloc[: , 0], color="r", lw=1 ,alpha = 0.7 ,shade=True,  ax = axes , label = 'Kde for ' + str(Binary_value[0]))
        sns.kdeplot(df_cat1.iloc[: , 0], color="b", lw=1 ,alpha = 0.7 ,shade=True,  ax = axes , label = 'Kde for ' + str(Binary_value[1]))
        
        axes.grid()
        axes.legend()
        axes.set_title('Differ ' + new_data.columns[1])
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + ' compare ' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()
    
    # 一个数值型一个多分类型
    elif(eq(new_class , [0 , 2])):
        if figsize == None : figsize = (8 , 8)
        
        fig , axes = plt.subplots(2 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化

        Numerical_autoplt(new_data.iloc[: , 0] , ax1 = axes[0][0] , ax2 = axes[0][1]) # 数值型的两个图
        Multitpye_autoplt(new_data.iloc[: , 1] , ax1 = axes[1][0] , ax2 = axes[1][1]) # 多分类的两个图

        plt.tight_layout() # 自动调整子图间距
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()

        # 绘制统一大图下，不同多分的各自类别的密度曲线。
        fig , axes = plt.subplots(figsize = (8,8) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Multi_value = new_data.iloc[: , 1].unique() # 获取这个多分类变量的各个值
        colors = sns.husl_palette(len(Multi_value)) # 获取对应这个长度的不同颜色盘
        # print(len(Multi_value) , len(colors))   此处供给于debug，如果出现了调色板不够用的时候，请查看长度是否一致
        # print(colors)
        for index , value in enumerate(Multi_value): # 根据各个类别绘制对应类别的核密度曲线
            sns.kdeplot(new_data[new_data.iloc[: , 1] == value].iloc[: , 0] , color = colors[index] ,shade=True , alpha = 0.7 ,  ax = axes , label = 'Kde for ' + str(value))
        
        axes.grid()
        axes.legend()
        axes.set_title('Differ '+ new_data.columns[1])
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + ' compare ' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()
    
    # 两个二分变量
    elif(eq(new_class , [1,1])):
        if figsize == None : figsize = (12 , 4)
        
        fig , axes = plt.subplots(1 , 3 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Binary_autoplt(new_data.iloc[: , 0] , ax1 = axes[0])
        Binary_autoplt(new_data.iloc[: , 1] , ax1 = axes[1])
        
        Nesting_pie(new_data , ax = axes[2]) # 复式饼图
        axes[2].set_xlabel("Inner:"+ new_data.columns[0])
        plt.tight_layout() # 自动调整子图间距
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()

    # 一个二分变量一个多分变量
    elif(eq(new_class , [1 , 2])):
        if figsize == None : figsize = (12 , 4)
        
        fig , axes = plt.subplots(1 , 3 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Multitpye_autoplt(new_data.iloc[: , 1] , ax2 = axes[0])
        Nesting_pie(new_data , ax = axes[1]) # 二分变量内环
        axes[1].set_xlabel("Inner:"+new_data.columns[0])
        Nesting_pie(new_data.iloc[: , [1 , 0]] , ax = axes[2]) # 多分变量内环
        axes[2].set_xlabel("Inner:"+new_data.columns[1])
        
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()

    # 2个多分类变量
    elif(eq(new_class , [2, 2])):
        if figsize == None : figsize = (8 , 8)
        
        fig , axes = plt.subplots(2 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Multitpye_autoplt(new_data.iloc[: , 0], ax1 = axes[0][0], ax2=axes[0][1])
        Multitpye_autoplt(new_data.iloc[: , 1], ax1=axes[1][0], ax2 = axes[1][1])
        plt.tight_layout() # 自动调整子图间距
        
        if save : plt.savefig(dir_path + '/' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()
        
        fig , axes = plt.subplots(figsize = (8,8) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Mosaic_plt(new_data , ax = axes)
        
        if save : plt.savefig(dir_path + '/Mosaic ' + new_data.columns[0] + '.' + new_data.columns[1] + '.png' , dpi = dpi)
        if show : plt.show()
    else:
        raise ValueError('The index is wrong \n 输入的index有误,请检查')

def Three_dim_autoplt(data , dataclass , filepath = None , save = True , show = False , figsize = None , dpi = None):
    """
    对一个二维数据进行描述性统计图绘制,并将结果结构化输出为图片保存到既定文件路径。
    Args:
        data : dataframe 应为3 维
        dataclass : list , 存储着对应位置的数据的类别, 具体为：数值型 0 ; 二分类型 1 ; 多分类型 2
        filepath : 想要保存到的文件夹路径 Defaults to None.
        save : 是否保存为.png , 默认为True
        show : 是否显示,默认为False 因为会阻塞进程
    """
    # 建立文件路径
    time = datetime.now()
    
    current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
    if filepath == None:
        dir_path = './descriptive_result_output/'+current_time
    else:
        dir_path = filepath
    if not os.path.exists(dir_path):  # 如果不存在该文件夹，则创建该文件夹。
        os.makedirs(dir_path)
    
    if dpi == None : dpi = 100


    new_data , new_class = data_sort(data , dataclass) # 对原data依照dataclass排序并获得排序后的data和class
    col_names = new_data.columns.to_list()
    
    # 3个数值型变量
    if(eq(new_class , [0 , 0 , 0])):
        # 首先绘制各自的一维图，合并成一个大图
        if figsize == None : figsize = (8 , 12)
        
        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        for i in range(3):
            Numerical_autoplt(new_data[col_names[i]] , ax1 = axes[i][0] , ax2 = axes[i][1])
        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/ ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 随后绘制矩阵散点图
        sns.pairplot(new_data , kind='reg',diag_kind='hist' , plot_kws = {'scatter_kws' : {'alpha' : 0.6 , 's': 7},'line_kws':{'color' : (232/255, 59/255, 12/255 , 0.5) , 'linewidth' : 1}} )
        fig = plt.gcf()
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        if save : plt.savefig(dir_path + '/Matrix_graph' + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 绘制3元正态的卡方qq图
        fig , axes = plt.subplots(figsize = (5 , 5) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        chi2_QQ(new_data , axes = axes)
        axes.set_title('Multi chi2 QQ plot')
        
        if save : plt.savefig(dir_path + '/chi2 QQ_plot' + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 绘制热力图
        fig , axes = plt.subplots(figsize = (5 , 5) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        sns.heatmap(new_data.corr() , ax = axes)
        axes.set_title('HeatMap')
        
        if save : plt.savefig(dir_path + '/HeatMap' + '.png' , dpi = dpi)
        if show : plt.show()
    
    # 2个数值型一个分类型(分为二分和多分类)
    
    elif(eq(new_class , [0 , 0 , 1]) or eq(new_class , [0 , 0 , 2])):
        # 首先绘制各自的一维图，合并成一个大图
        if figsize == None : figsize = (8 , 12)
        
        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        for i in range(2):
            Numerical_autoplt(new_data[col_names[i]] , ax1 = axes[i][0] , ax2 = axes[i][1])
        
        if(new_class[-1] == 1): # 如果是2分类则绘制2分类的图
            Binary_autoplt(new_data[col_names[2]] , ax1 = axes[2][0] , ax2 = axes[2][1])
        else: # 如果是多分类则绘制多分类的图
            Multitpye_autoplt(new_data[col_names[2]] , ax1 = axes[2][0] , ax2 = axes[2][1])
        
        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/ ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
            
        # 绘制各数值变量的不同分类下的kde
            
        fig , axes = plt.subplots(1 , 2 , figsize = (10 , 5) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            
        Multi_value = new_data.iloc[: , 2].unique() # 获取这个多分类变量的各个值
        colors = sns.husl_palette(len(Multi_value)) # 获取对应这个长度的不同颜色盘
        # print(len(Multi_value) , len(colors))   此处供给于debug，如果出现了调色板不够用的时候，请查看长度是否一致
        # print(colors)
        for index , value in enumerate(Multi_value): # 根据各个类别绘制对应类别的核密度曲线
            sns.kdeplot(new_data[new_data.iloc[: , 2] == value].iloc[: , 0] , color = colors[index] ,shade=True , alpha = 0.7 ,  ax = axes[0] , label = 'Kde for ' + str(value))
            sns.kdeplot(new_data[new_data.iloc[: , 2] == value].iloc[: , 1] , color = colors[index] ,shade=True , alpha = 0.7 ,  ax = axes[1] , label = 'Kde for ' + str(value))
        for i in range(2):
            axes[i].grid()
            axes[i].legend()
            axes[i].set_xlabel(col_names[2])
            axes[i].set_ylabel(col_names[i])
            axes[i].set_title('Differ kde')

        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/diff_kde of ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 绘制二维散点图，用不同类别做颜色标注
        
        fig , axes = plt.subplots(figsize = (10 , 10) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        sns.scatterplot(data=new_data, x=col_names[0], y=col_names[1], hue=col_names[2], palette=colors , ax = axes , s = 8 , alpha = 0.7)
        
        if save : plt.savefig(dir_path + '/scatter of ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
    
    # 1数值2个分类
    
    elif(eq(new_class , [0 , 1 , 1]) or eq(new_class , [0 , 1 , 2]) or eq(new_class , [0 , 2 , 2])): 
        # 首先绘制各自的一维图，合并成一个大图
        if figsize == None : figsize = (8 , 12)
        
        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        Numerical_autoplt(new_data[col_names[0]] , ax1 = axes[0][0] , ax2 = axes[0][1])
        for i in range(1,3):
            if(new_class[i] == 1):
                Binary_autoplt(new_data.iloc[: , i] , ax1 = axes[i][0] , ax2 = axes[i][1])
            else:
                Multitpye_autoplt(new_data.iloc[: , i] , ax1 = axes[i][0] , ax2 = axes[i][1])
        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/ ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()

        # 绘制数值变量在各个分类变量下的kde
        Multi_value1 = new_data.iloc[: , 1].unique()
        Multi_value2 = new_data.iloc[: , 2].unique()
        colors1 = sns.husl_palette(len(Multi_value1)) # 获取第二列对应这个长度的不同颜色盘
        colors2 = sns.husl_palette(len(Multi_value2)) # 获取第三列对应这个长度的不同颜色盘
        fig , axes = plt.subplots(1 , 2 , figsize = (10 , 5) , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        for index , value in enumerate(Multi_value1): # 第一个分类变量对应的kde
            sns.kdeplot(new_data[new_data.iloc[: , 1] == value].iloc[: , 0] , color = colors1[index] ,shade=True , alpha = 0.7 ,  ax = axes[0] , label = 'Kde for ' + str(value))
        for index , value in enumerate(Multi_value2): # 第二个分类变量对应的kde
            sns.kdeplot(new_data[new_data.iloc[: , 2] == value].iloc[: , 0] , color = colors2[index] ,shade=True , alpha = 0.7 ,  ax = axes[1] , label = 'Kde for ' + str(value))
        for i in range(2):
            axes[i].grid()
            axes[i].legend()
            axes[i].set_xlabel(col_names[i+1])
            axes[i].set_ylabel(col_names[0])
            axes[i].set_title('Differ kde')
        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/diff_kde of ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
        
        # 绘制两分类变量之间对的关系
        
        if(eq(new_class[1:] , [1 , 1])): # 如果是两个二分变量
            fig , axes = plt.subplots(figsize = (5 , 5) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            Nesting_pie(new_data.iloc[: , [1 , 2]] , ax = axes) # 绘制嵌套饼图
            if save : plt.savefig(dir_path + '/Pie ' + '.'.join(col_names[1:]) + '.png' , dpi = dpi)
            if show : plt.show()
        elif(eq(new_class[1:] , [1 , 2])): # 如果为一个二分一个多分
            fig , axes = plt.subplots(1,  2 , figsize = (10 , 5) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            Nesting_pie(new_data.iloc[: , [1 , 2]] , ax = axes[0]) # 绘制嵌套饼图 内环为2分变量
            Nesting_pie(new_data.iloc[: , [2 , 1]] , ax = axes[1]) # 绘制嵌套图 内环为多分变量
            axes[0].set_xlabel('Inner is '+ col_names[1])
            axes[1].set_xlabel('Inner is '+ col_names[2])
            plt.tight_layout() # 自动调整子图间距
            if save : plt.savefig(dir_path + '/Pie ' + '.'.join(col_names[1:]) + '.png' , dpi = dpi)
            if show : plt.show()
        else: # 如果是2个多分类变量
            fig , axes = plt.subplots(1,  2 , figsize = (10 , 5) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            Mosaic_plt(new_data.iloc[: , [1 , 2]] , ax = axes[0]) # 绘制嵌套饼图 内环为2分变量
            Mosaic_plt(new_data.iloc[: , [2 , 1]] , ax = axes[1]) # 绘制嵌套图 内环为多分变量
            plt.tight_layout() # 自动调整子图间距
            if save : plt.savefig(dir_path + '/Mosaic ' + '.'.join(col_names[1:]) + '.png' , dpi = dpi)
            if show : plt.show()
    
    # 3个分类型变量        
    
    else : 
        # 首先绘制各自的一维图，合并成一个大图
        if figsize == None : figsize = (8 , 12)
        
        fig , axes = plt.subplots(3 , 2 , figsize = figsize , dpi = dpi)
        fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
        
        for i in range(3):
            if(new_class[i] == 1):
                Binary_autoplt(new_data.iloc[: , i] , ax1 = axes[i][0] , ax2 = axes[i][1])
            else:
                Multitpye_autoplt(new_data.iloc[: , i] , ax1 = axes[i][0] , ax2 = axes[i][1])
        
        plt.tight_layout() # 自动调整子图间距
        if save : plt.savefig(dir_path + '/ ' + '.'.join(col_names) + '.png' , dpi = dpi)
        if show : plt.show()
        
        class1 = new_class[:2] ; class2 = new_class[::2] ; class3 = new_class[1:]  # C_3^2一下，出3个子集绘图
        col1 = col_names[:2] ; col2 = col_names[::2] ; col3 = col_names[1:]
        class_list = [class1 , class2 , class3] ; col_list = [col1 , col2 , col3]
        
        
        for unit , col in zip(class_list , col_list):
            if eq(unit , [2 , 2]):
                fig , axes = plt.subplots(1,  2 , figsize = (12 , 6) , dpi = dpi)
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                Mosaic_plt(new_data[col] , ax = axes[0])
                Mosaic_plt(new_data[col[::-1]] , ax = axes[1])
                plt.tight_layout() # 自动调整子图间距
                if save : plt.savefig(dir_path + '/Mosaic ' + '.'.join(col) + '.png' , dpi = dpi)
                if show : plt.show()
                
            else:
                fig , axes = plt.subplots(1,  2 , figsize = (10 , 5) , dpi = dpi)
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                Nesting_pie(new_data[col] , ax = axes[0]) # 绘制嵌套饼图 内环为前一列变量
                Nesting_pie(new_data[col[::-1]] , ax = axes[1]) # 绘制嵌套图 内环为后一列变量
                axes[0].set_xlabel('Inner is '+ col[0])
                axes[1].set_xlabel('Inner is '+ col[1])
                plt.tight_layout() # 自动调整子图间距
                if save : plt.savefig(dir_path + '/Pie ' + '.'.join(col) + '.png' , dpi = dpi)
                if show : plt.show()
        
        # 绘制3个变量的3维列联表并输出为.csv文件
        
        temp_df = pd.DataFrame(new_data.groupby(col_names).size().reset_index()) # 生成对应的列联表对应的dataframe
        temp_df.columns=col_names + ['val'] # 生成对应的列名
        
        three_dim_cross_tab = pd.pivot_table(temp_df, values= 'val', index=col_names[:2], columns=col_names[2],
                                                aggfunc=np.sum , margins = True).fillna(0)
        if save : three_dim_cross_tab.to_csv(dir_path + '/crosstab.csv')
        if show : print(three_dim_cross_tab)
        
        
        
def Auto_plt(data , dataclass , filepath = None):
    """
    终极函数,融合了一维,二维,三维情况下的各种分支,全自动输入数据和数据类型以及要保存的文件位置，生成

    Args:
        data (dataframe): 初步清洗过后的dataframe
        dataclass (list) : 对应data的数据类型的一个原生python list
        filepath (str) : 要保存的文件位置,默认为None,系统自动生成
    """
    dpi = 100  
    
    new_data , new_class = data_sort(data , dataclass) # 对原data依照dataclass排序并获得排序后的data和class
    n = len(dataclass)
    if n == 1:
        raise ValueError('It\'s only one dim , plot it by yourself please')
    elif  n == 2 :
        Two_dim_autoplt(new_data , new_class , filepath = filepath)
    elif  n == 3 :
        Three_dim_autoplt(new_data , new_class , filepath = filepath)
    else:
        
        col_names = new_data.columns # 获取列名
        
        time = datetime.now() ; current_time = f'{time.month}_{time.day}_{time.hour}_{time.minute}' # 记录当前的系统时间
        if filepath == None:
            dir_path = './descriptive_result_output/'+current_time
        else : dir_path = filepath +'/' +  current_time
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
        num_len = len(numerical_class) ; type_len = len(type_class) # 获取数值型数据的个数以及分类型数据的个数

        if num_len == 0: pass
        elif num_len == 1:
            fig , axes = plt.subplots(1,  2 , figsize = (8 , 4) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            Numerical_autoplt(new_data.iloc[:,0] , ax1 = axes[0] , ax2 = axes[1])
            plt.savefig(dir_path + '/'+col_names[0]+'.png' , dpi = dpi)
        elif num_len == 2:
            Two_dim_autoplt(new_data[col_names[:2]] , dataclass = [0 , 0] , filepath=dir_path)
        elif num_len == 3:
            Three_dim_autoplt(new_data[col_names[:3]] , dataclass = [0 , 0 , 0] , filepath = dir_path)
        else:
            for i in range(num_len): # 每个维度的变量都要作图
                fig , axes = plt.subplots(1,  2 , figsize = (8 , 4) , dpi = dpi)
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                Numerical_autoplt(new_data.iloc[: , i] , ax1 = axes[0] , ax2 = axes[1])
                plt.savefig(dir_path + '/'+col_names[i]+'.png' , dpi = dpi)
            if num_len <= 10: # 可以绘制矩阵散点图
                sns.pairplot(new_data.iloc[: , :num_len] , kind='reg',diag_kind='hist' , plot_kws = {'scatter_kws' : {'alpha' : 0.6 , 's': 7},'line_kws':{'color' : (232/255, 59/255, 12/255 , 0.5) , 'linewidth' : 1}} )
                fig = plt.gcf()
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                plt.savefig(dir_path + '/Matrix_graph' + '.png' , dpi = dpi)
            if num_len <=30: # 可以绘制热力图
                fig , axes = plt.subplots(figsize = (10 , 10) , dpi = dpi)
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                sns.heatmap(new_data.iloc[: , :num_len].corr() , ax = axes)
                axes.set_title('HeatMap')
                plt.savefig(dir_path + '/HeatMap' + '.png' , dpi = dpi)
            
            # 绘制卡方qq图
            fig , axes = plt.subplots(figsize = (6 , 6) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            chi2_QQ(new_data[col_names[:num_len]] , axes = axes)
            axes.set_title('Multi chi2 QQ plot')
            plt.savefig(dir_path + '/Chi2 QQ' + '.png' , dpi = dpi)
        
        if type_len == 0 : pass
        elif type_len == 1: 
            fig , axes = plt.subplots(1,  2 , figsize = (8 , 4) , dpi = dpi)
            fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
            if type_class[0] == 1:
                Binary_autoplt(new_data.iloc[: , num_len] , ax1 = axes[0] , ax2 = axes[1])
            else :
                Multitpye_autoplt(new_data.iloc[: , num_len] , ax1 = axes[0] , ax2 = axes[1])
            plt.savefig(dir_path + '/'+col_names[num_len]+'.png' , dpi = dpi)
        elif type_len == 2:
            Two_dim_autoplt(new_data.iloc[: , num_len:num_len+2] , dataclass = type_class , filepath = dir_path)
        elif type_len == 3:
            Three_dim_autoplt(new_data.iloc[: , num_len:num_len+3] , dataclass = type_class , filepath = dir_path)
            
        # 当分类型变量的数量超过4时，仅仅绘制其自身的图片，如果需要研究更多相关性，建议三维或者二维图片.
        else:
            for index , i in enumerate(type_class):
                fig , axes = plt.subplots(1,  2 , figsize = (8 , 4) , dpi = dpi)
                fig.patch.set_facecolor("white") #* 设置背景 以免保存的图片背景虚化
                if i == 1:
                    Binary_autoplt(new_data.iloc[: , num_len + index] , ax1 = axes[0] , ax2 = axes[1])
                else :
                    Multitpye_autoplt(new_data.iloc[: , num_len + index] , ax1 = axes[0] , ax2 = axes[1])
                plt.savefig(dir_path + '/'+col_names[num_len + index]+'.png' , dpi = dpi)
                


if __name__ == '__main__':
    df1 = pd.read_csv("data/select_data.csv",encoding = "utf-8")
    df2 = pd.read_csv('data/label_train.csv',index_col=0)
    index = df2.index
    pre_data = df1.loc[index , ].join(df2['MATH'])
    pre_data = pre_data.drop(pre_data.columns[0] , axis=1)
    pre_data = pre_data[['MATH']+list(pre_data.columns[0:-1].values)]
    pre_data = pre_data.head(600).reset_index(drop=True)
    
    df3 = pd.read_csv('data/test_data.csv' , encoding = "utf-8")
    
    
    
    # fig , axes = plt.subplots(1 , 2 , figsize = (10 , 5) , dpi = 150)
    
    # Binary_autoplt(df3['livingroom'] , ax1 = axes[0] , ax2 = axes[1])
    # Numerical_autoplt(pre_data['MATH'] , ax1 = axes[0] , ax2 = axes[1])
    # Multitpye_autoplt(df3['region'] , ax1 = axes[0] , ax2 = axes[1])
    
    Auto_plt(df3[['livingroom' , 'bathroom' , 'bedroom']] , [1 , 1 , 2])
    
    