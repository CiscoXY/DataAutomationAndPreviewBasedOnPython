# 6.7更新 Final version:

## 该任务主要分为以下5个组份(组件位于Analyzation_relative中)：

* 1. Data_process, 数据处理，包含**缺失值**和**异常值**
* 2. Descriptive_statistics, 描述性统计
* 3. Multivariate_statistical, 多元统计分析，主要为PCA
* 4. Regression, 回归
* 5. Statistical_inference, 统计推断，只包含正态性检验

## 具体用法简介(详情参考函数备注以及.py文件内参数解释)

* 1. Data_process中具体用法参考源文件
* 2. Descriptive_statistics, 如果想绘制**一维**的描述性统计图，可以使用Numeric_autoplt/Binary_autoplt/Multitype_autoplt  
对于≥2维的数据，直接使用**Auto_plt**即可  
更多可定制化图形详见文件内各函数描述
* 3. Multivariate_statistical具体用法参考源文件，重点为**PCA(可提供可视化)**
* 4. Regression, 核心组件为Stepwise_reg，如果想图省事直接Automatic_reg即可。其余异常检验方法参考函数描述。
* 5. Statistical_inference, 由于只包含正态性检验，直接输入数据即可，具体大样本小样本会自动判断。