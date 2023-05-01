import matplotlib as mpl
import matplotlib.pyplot as plt
#*from matplotlib.patches import Ellipse , Rectangle
import numpy as np

import pandas as pd
from scipy.stats import f , t 
from scipy.linalg import svd
#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)


def Mahalanobis_Distance(X):
    '''
    计算X的马氏距离,X应为n*p的矩阵,而不是数据框,p个变量,样本容量为n,返回一个1*p的向量存储dist
    '''
    n , m = X.shape
    S = np.cov(X.T) #* 求出样本协方差矩阵
    mu = np.average(X , axis = 0) #* 按列求均值
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = np.matmul(np.matmul(X[i] - mu , np.linalg.inv(S)) , (X[i] - mu).T)
    return dist

def T_2_test(X , mu_0):
    '''
    返回X对应mu_0的T方统计量的值
    '''
    n , m = X.shape
    S = np.cov(X.T) #*求出样本协方差矩阵
    bar_x = np.average(X , axis = 0)
    T_2 = n * np.matmul( np.matmul(bar_x - mu_0 , np.linalg.inv(S)) , (bar_x - mu_0).T)
    return T_2 #* 返回对应的T^2统计量

def Pang_region(df , alpha = 0.05):
    '''
    df为需要构造庞弗罗尼置信区间的原始数据 nxp的数据框,p维待估计参数
    alpha为显著水平,默认为0.05
    返回一个pd.dataframe形式的数据框,记录p维参数的对应置信下上限
    '''
    X = df.values
    n , p = X.shape ; S = np.cov(X.T) ; mu = np.average(X , axis = 0)
    params= t.isf(alpha/(2*p) , n-1)#* 显著性水平alpha 右侧分位数 默认m = p
    inf_pang = [] ; sup_pang = []
    a = np.identity(p)
    for i in range(p):
        inf_pang.append(np.matmul(a[i] , mu.T) - params * np.sqrt(np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
        sup_pang.append(np.matmul(a[i] , mu.T) + params * np.sqrt(np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
    ColNames_List = df.columns.tolist() #* 获取数据框的列名
    region = pd.DataFrame({'置信下限':inf_pang , '置信上限': sup_pang} , index=ColNames_List)
    return region

def T2_region(df , alpha = 0.05):
    '''
    df为需要构造T2置信区间的原始数据 nxp的数据框,p维待估计参数
    alpha为显著水平,默认为0.05
    返回一个pd.dataframe形式的数据框,记录p维参数的对应置信下上限
    '''
    X = df.values
    n , p = X.shape ; S = np.cov(X.T) ; mu = np.average(X , axis = 0)
    F_alpha = f.isf(alpha , p , n-p) #* 显著性水平alpha = 0.05
    params_1 = p*(n-1)/(n-p) * F_alpha
    inf_t2 = [] ; sup_t2 = []
    a = np.identity(p)
    for i in range(p):
        inf_t2.append(np.matmul(a[i] , mu.T) - np.sqrt(params_1 * np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
        sup_t2.append( np.matmul(a[i] , mu.T) + np.sqrt(params_1 * np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
    ColNames_List = df.columns.tolist() #* 获取数据框的列名
    region = pd.DataFrame({'置信下限':inf_t2 , '置信上限': sup_t2} , index=ColNames_List)
    return region



def PCA(df , use_cor = True , percentage = 0.85 , ax = None):   #  搞定了！
    '''
    传入数据框df,
    默认使用协方差矩阵进行PCA分解,如果use_cor = False那么就会使用协方差矩阵进行分解
    传入希望的主成分权重(主成分对应特征根和/总特征根和),默认0.85,需要0<persentage<1
    standard = 共线性性的判断标准,即求解的最小特征值小于0.3
    ax为传入的可供绘图的matplotlib子图,如果值为非None,那么在ax上绘制累计方差贡献率的帕累托图
    '''
    # Check df is a dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df should be a dataframe')
    
    if use_cor:
        cov_matrix = df.corr()
    else:
        cov_matrix = df.cov()
    # Perform SVD on the cov_matrix 
    U, S, V = svd(cov_matrix)
    print(S)
    # Calculate the explained variance ratio and eigenvectors
    variance_explained = S/np.sum(S) # 计算各特征向量的方差贡献率
    cumulative_explained = variance_explained #计算累计方差贡献率
    
    for i in range(1, len(S)):
        cumulative_explained[i] = cumulative_explained[i-1] +variance_explained[i]
    cumulative_explained = np.round(cumulative_explained , 3)
    if ax is not None:
        x = range(1 , len(S)+1)
        ax.bar(x , cumulative_explained , color= (240/255, 154/255, 74/255 , 0.5) , width = 0.5)
        ax.plot(x , cumulative_explained , '-o' , label = 'Cumulative Explained Variance')
        ax.set_xlabel('累计主成分数量') 
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_xticks(x)
        for x, y in zip(x, cumulative_explained):
            ax.text(x, y-0.05, str(y), ha='center')
        ax.legend(loc = 'best')
        ax.set_title('方差累计贡献率')
        ax.grid()
    
    
    for index , i in enumerate(cumulative_explained):
        if i >= percentage:
            return V.T[:,:index+1] # 选取满足累计方差解释率到persentage的index个主成分对应的特征向量并返回



def CCA_select(data_X , data_Y , cca_num , is_cov = False , Cov = None):
    '''
    传入样本数据X和数据Y,均为pandas的dataframe形式,给定要的典型相关变量个数,输出对应的相关系数与相关变量的线性变换向量
    如果传入的是协方差矩阵,则data_X为数据X的变量数p,data_Y为数据Y的变量数q,is_cov应为True,Cov为传入的协方差矩阵
        对应输出为对应的典型相关系数和对应的相关变量的线性变换向量
    注:返回的向量矩阵为每一行对应一个线性变换变量，而不是每一列对应一个
    Args: 
        data_X : The sample data of sample X
        data_Y : The sample data of sample Y
        cca_num : The expected number of cca
        is_cov : If you wanna input the cov
        Cov : The cov you input , must be an numpy.array with 2 dim
    returns:
        if 'is_cov' is False , which is the default setting you will receive the normal result
        else , you will receive the cca_corr and it's eigenvector , which is the vector of the linear transfer of original data
    '''
    if(is_cov): # 当传入协方差矩阵时
        if(Cov is None):
            print("You must place the Cov!")
            exit(-1)
        p = data_X ; q = data_Y
        if(p > q):
            print("dim(X)p= must <= dim(Y)=q")
            exit(-2)
    else:
        n , p = data_X.shape ; n , q = data_Y.shape # 获取对应的维数
        if(p > q):
            print("dim(X)p= must <= dim(Y)=q")
            exit(-2)
        Join_data = data_X.join(data_Y)
        Cov = Join_data.cov().values # 获取协方差矩阵
    cov11 = Cov[0:p , 0:p] ; cov12 = Cov[0:p , p:p+q]
    cov21 = Cov[p:p+q , 0:p] ; cov22 = Cov[p:p+q , p:p+q] # 分割协方差矩阵
    cov22_reverse = np.linalg.inv(cov22)
        
    T_a = np.dot(np.dot(np.dot(np.linalg.inv(cov11) , cov12) , cov22_reverse) , cov21) # 求出线换a对应的矩阵
    lambda_a , eigen_a = np.linalg.eig(T_a)  # 特征值总计p个，所以下面的右侧只能取前p行
    temp_matrix = np.dot(np.dot(cov22_reverse,cov21),eigen_a) # 没有独特意义,qxp的中间矩阵 
    eigen_b =  np.dot(np.diag(lambda_a ** -0.5) , temp_matrix.T) # 获得了原始b的矩阵B
    eigen_a = eigen_a.T# 并且转置一下，这样第一行对应第一个特征向量
        
    lambda_reverse = np.sort(lambda_a)[::-1] ; lambda_reverse_index = np.argsort(lambda_a)[::-1]
    result_lambda = lambda_reverse[0:cca_num]**0.5 # 获取前k个典型相关系数
    A = eigen_a[lambda_reverse_index[:cca_num]] # 第一步获取的没有正规化的矩阵A
    B = eigen_b[lambda_reverse_index[:cca_num]] # 第一步获取的没有正规化的矩阵B
        
    regular_matrix_a = np.diag(np.sqrt(1/np.diagonal(np.dot(np.dot(A , cov11),A.T)))) # 获取a'Sigma a的对角元的倒数的开根对应的对角阵,pxp
    regular_matrix_b = np.diag(np.sqrt(1/np.diagonal(np.dot(np.dot(B , cov22),B.T)))) # 获取b'Sigma b的对角元的倒数的开根对应的对角阵,qxq
        
    result_A = np.dot(regular_matrix_a , A) # 正规化A
    result_B = np.dot(regular_matrix_b , B) # 正规化B
        
    return result_lambda , result_A , result_B # 返回所要的典型相关系数和对应的变换阵A和B

def Fisher_linear_discriminant(X , Y , Z , criterion_params = 1):
    '''
    传入总体X,Y和待判别的数据Z,均为dataframe形式
    需求这三个数据均为正态分布,criterion_params是最小ECM法则不等式右侧的ln里面的系数
    默认值是1
    返回矩阵Z在这个Fisher线性判别函数下的分类情况,是一个维数为1xn的np.array,n为矩阵Z的长度(也就是行数)
    '''
    # 首先检验输入的矩阵是否符合要求
    n1 , p1 = X.shape ; n2 , p2 = Y.shape ; n , p = Z.shape
    if(p1 != p2 and p1 != p):
        print('输入的三总体的数据维数必须相同')
        exit(-1)
    S1 = X.cov().values ; S2 = Y.cov().values # 获取两个总体各自的样本协方差矩阵
    Sp = ((n1-1)*S1 + (n2-2)*S2)/(n1+n2-2) # 获取Sp这个协方差矩阵
    Sp_inv = np.linalg.inv(Sp) # 获取这个矩阵的逆
    x1_bar = np.average(X.values , axis=0) # 计算X均值向量
    y1_bar = np.average(Y.values , axis=0) # 计算Y均值向量
    m = 0.5 * np.dot(np.dot(np.array([x1_bar - y1_bar]) , Sp_inv) , np.array([x1_bar + y1_bar]).T) # 获取参数m
    if(Z.values.ndim == 1):
        copy_Z = np.array(Z.values) # 如果维数只有一维，那么就必须转换成高维 方便进行矩阵运算
    else: 
        copy_Z =  Z.values
    linear_values = np.dot(np.dot(np.array([x1_bar - y1_bar]) , Sp_inv) , copy_Z.T) - m # 获得计算后的线性判别函数的值
    Judge_array = linear_values >= np.log(criterion_params) # 如果大于等于 则分配到X所属类 反之分类到Y所属类
    return Judge_array # 返回一个bool类型的数组，True为X所属类，False为Y所属类




if __name__=="__main__":
    df = pd.read_table("data/T5-1.dat", sep="\s+",header = None)
    # df = pd.read_csv("data/test_data.csv",encoding = "utf-8")
    # df = df.loc[df["region"] == "朝阳" , ['rent' , 'area' , 'room' , 'subway']]

    fig , axes = plt.subplots(1 ,1 , figsize = (6 ,6) , dpi = 100)
    eng_v = PCA(pd.DataFrame(df) ,use_cor=False ,  ax = axes , percentage = 0.99)
    print(eng_v)
    print(np.cov((df @ eng_v).T))
    plt.show()