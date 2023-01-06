# encoding:utf-8

import random
import numpy as np
import json
import pandas as pd
import sys,os
sys.path.append('C:/Users/xiaoyaohou/Documents/Model/TOOL/Profile')
import xlsxwriter
import application.report_percentile
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']= False    # 正常显示负号
from config import r

'''
BP神经网络Python实现
'''


def sigmoid(x):
    # 激活函数
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    # 激活函数的导数
    return sigmoid(x) * (1 - sigmoid(x))

class BPNNRegression:
    
    def __init__(self, sizes=[1,3,1]):

        # 神经网络结构
        # 格式： [2,3,1] 表示 2个输入变量，3个隐层单元，1个输出
        self.sizes = sizes

        # 初始化偏差，除输入层外， 其它每层每个节点都生成一个 biase 值（0-1）
        self.biases = [np.random.randn(n, 1) for n in sizes[1:]]
        
        # 随机生成每条神经元连接的 weight 值（0-1）
        # 注意这里的权重矩阵的格式，列数为
        
        self.weights = [np.random.randn(r, c)  for c, r in zip(sizes[:-1], sizes[1:])]

        self.state = {'in':[],'out':[]}
        self.input = None
        self.output = None
        self.err_epoch = []
        
    def upload(self,path):

        # 加载保存的神经网络
        # 输入：json形式的神经网络

        path_all = os.path.join(os.getcwd(),path)
        assert (os.path.isdir(os.path.dirname(path_all))), '路径无效'
        f=open(path_all,'r',encoding='utf-8')
        info=json.load(f)
        f.close()

        self.sizes = info['sizes']
        self.biases = [np.array(biase) for biase in info['biases']]
        self.weights = [np.array(weight) for weight in info['weights']]
        self.epochs = info['epochs']
        self.mini_batch_size = info['mini_batch_size']
        self.eta = info['eta']
        self.error = info['error']

    def download(self,path):

        # 下载神经网络的参数结构
        # 输出: json形式的神经网络

        path_all = os.path.join(os.getcwd(),path)
        assert (os.path.isdir(os.path.dirname(path_all))), '路径无效'
        info = {}
        info['sizes'] = self.sizes
        info['biases'] = [biase.tolist() for biase in self.biases]
        info['weights'] = [weight.tolist() for weight in self.weights]
        info['epochs'] = self.epochs
        info['mini_batch_size'] = self.mini_batch_size
        info['eta'] = self.eta
        info['error'] = self.error
        f=open(path_all,'w',encoding='utf-8')
        json.dump(info,f,indent=4,ensure_ascii=False)
        f.close()
        
    def feed_forward(self, x):
    
        # 前向传输计算输出神经元的值
        # x: 输入

        self.input = x
        self.state['in'] = []
        self.state['out'] = []
        self.state['out'].append(x)
        for i, b, w in zip(range(len(self.biases)), self.biases, self.weights):
            self.state['in'].append(np.dot(w, self.state['out'][-1]) + b)
            self.state['out'].append(sigmoid(self.state['in'][-1]))
        self.state['out'].pop()
        self.output = self.state['in'][-1]
        
#**************************** MBGD（开始） **************************** 

    def MSGD(self, training_data, epochs = 3000, mini_batch_size = 1, eta = 0.1, error = 0.01, task=None, project_name = None):
    
        # 小批量随机梯度下降法
        # eta: 学习速率
        # error：误差阈值

        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.eta = eta
        self.error = error
        
        # 样本个数
        n = len(training_data) 
        
        for j in range(epochs):
            r.set(f'{project_name}_{task}', j / epochs)
            if (j == epochs - 1):
                r.set(f'{project_name}_{task}', 1)
        
            # 模拟退火因子
            ratio = 1- j*1.0/epochs
            
            # 随机打乱训练集顺序
            random.shuffle(training_data)
            
            # 根据小样本大小划分子训练集集合
            mini_batchs = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            
            # 利用每一个小样本训练集更新 w 和 b
            for mini_batch in mini_batchs:
                self.updata_WB_by_mini_batch(mini_batch, eta, ratio)
            
            #迭代一次后结果
            self.err_epoch.append(self.evaluate(training_data))
            
            # 每1000次输出一次结果
            if j%1000 == 0:
                print("Epoch {0} Error {1}".format(j, self.err_epoch[-1]))
            if self.err_epoch[-1] < error:
                break

        return self.err_epoch

#**************************** MBGD（结束） **************************** 


    def updata_WB_by_mini_batch(self, mini_batch, eta, ratio):
    

        # 利用小样本训练集更新 w 和 b
        # mini_batch: 小样本训练集
        # eta: 学习率

        # 创建存储迭代小样本得到的 b 和 w 偏导数空矩阵，大小与 biases 和 weights 一致，初始值为 0
        batch_par_b = [np.zeros(b.shape) for b in self.biases]
        batch_par_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # 根据小样本中每个样本的输入 x, 输出 y, 计算 w 和 b 的偏导
            delta_b, delta_w = self.back_propagation(x, y)
            
            # 累加偏导 delta_b, delta_w 
            batch_par_b = [bb + dbb for bb, dbb in zip(batch_par_b, delta_b)]
            batch_par_w = [bw + dbw for bw, dbw in zip(batch_par_w, delta_w)]
            
        # 根据累加的偏导值 delta_b, delta_w 更新 b, w
        # 由于用了小样本，因此 eta 需除以小样本长度
        
#**************************** SAA代码（开始） **************************** 

        self.weights = [w - (eta / len(mini_batch)*ratio) * dw
                        for w, dw in zip(self.weights, batch_par_w)]
        self.biases = [b - (eta / len(mini_batch)*ratio) * db
                        for b, db in zip(self.biases, batch_par_b)]

#**************************** SAA代码（结束） **************************** 

    def back_propagation(self, x, y):

        # 利用误差后向传播算法对每个样本求解其 w 和 b 的更新量
        # x: 输入神经元，行向量
        # y: 输出神经元，行向量
        

        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        # 1. 前向传播，更新隐层和输出层状态
        self.feed_forward(x)

        # 2. 计算目标和输出的误差（结果为求偏导后的结果）
        delta = -1.0 * self.cost_function(self.output, y)
        
        # 3. 更新隐层到输出层的偏置向量和权值矩阵
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, self.state['out'][-1].T) # delta: [n,1] 向量， self.state: [h,1] 向量，h: 最后一层隐层的数量
        
        # 4. 更新隐层内部的偏置向量和权值矩阵
        for lev in range(1, len(self.state['in'])):
            z = self.state['in'][-lev-1]
            zp = sigmoid_prime(z)
            delta = np.dot(self.weights[-lev].T, delta) * zp
            delta_b[-lev-1] = delta
            delta_w[-lev-1] = np.dot(delta, self.state['out'][-lev-1].T)

        return (delta_b, delta_w)
    

#**************************** 评价代码（开始） **************************** 

    def evaluate(self, train_data):
        # 用来评测模型的误差
        # train_data: 训练数据
    
        out = []
        obj = []
        for x, y in train_data: 
            self.feed_forward(x)
            out.append(self.output)
            obj.append(y)
        return RMSE_M(out,obj)[0]
    
#**************************** 评价代码（结束） **************************** 

    def predict(self, test_input):
        # 用来预测未知的样本
        
        predict_result = []
        for x in test_input:
            self.feed_forward(x)
            predict_result.append(self.output)
        return predict_result

    def cost_function(self, output_a, y):
        # 损失函数

        return (y - output_a)


'''
相关的函数
'''
def RMSE_M(pred, y_obj):

    # 此函数用来计算RMSE和RSE
    # pred： 预测值
    # y_obj: 真实值

    len_y = len(y_obj)
    len_pred = len(pred)
    assert (len_y == len_pred), '预测和输出长度不一样'
    rmse = np.sqrt(np.sum([np.sum((p-y)**2) for p,y in zip(pred,y_obj)])*1.0/len_y)
    rse = rmse/np.sqrt((np.sum(np.sum(y**2) for y in y_obj)/len_y))
    return (rmse, rse)
    

    
def df_to_list(df, ls_col):
    
    # 此函数把dataframe转为神经网络可以接收的列表格式
    # df： 输入源，其中应该包含 ls_col 中所有的字段
    # ls_col: 需要用到的数据字段，以list形式输入

    
    assert (set(ls_col).issubset(set(df.columns.tolist()))), '字段不在dataframe中'
    ls_array = []
    for loc, item in df[ls_col].iterrows():
        ls_array.append(np.array(item.values))
    return ls_array

def anti_normalize(ls_data, ls_col, max_col, min_col, med_col):
    

    # 此函数把归一化之后的数据通过逆归一化的方式得到原有的数据
    # ls_data: 需要进行逆归一化的数据矩阵列表
    # ls_col: 数据矩阵列表对应的字段列表
    # max_col: 每一数据字段的最大值
    # min_col: 每一数据字段的最小值
    # med_col: 每一数据字段的中位值

    len_data = len(ls_data)
    len_col = len(ls_col)
    ls_data_an = []
    assert (len_col == ls_data[0].shape[0]), '数据矩阵列表格式和字段列表不匹配'
    for d in ls_data:
        ls_data_an.append([d[c][0]*(max_col[ls_col[c]] - min_col[ls_col[c]])+med_col[ls_col[c]] for c in range(len_col)])
    return ls_data_an

'''
    神经网络学习
'''


if __name__=='__main__':
    # 本地路径
    
    path_cur = os.getcwd()
    
    # 数据读取
    
    path_data = 'data/312.csv'
    file_data = os.path.join(path_cur,path_data)
    # pd_file = pd.read_csv(file_data, sep =',',encoding= 'gb18030')
    try:
        pd_file = pd.read_csv(file_data, sep =',',encoding = 'utf-8')
    except Exception as e:
        print(e)
    finally:
        pd_file = pd.read_csv(file_data, sep =',',encoding= 'gb18030')
    
    
    # 输入输出变量
    
    # 菱香区间
    col_in = ['舱内土压', '推进速度', '[15]刀盘旋转速度']
    col_out = ['[28]総推力32bit', '[22]刀盘扭矩', '地表沉降']
    col_use = col_in + col_out
    
    
    # 基于人工信息的数据异常值处理
    pd_file_filter = pd_file.copy()[col_use]
    # pd_file_filter = pd_file_filter[pd_file_filter['千斤顶总推力'] > 10000] # 程博：千斤顶总推力正常应该是10000以上
    # pd_file_filter = pd_file_filter[pd_file_filter['推进速度'] > 1e-6] # 推进速度应该 > 0
    # pd_file_filter = pd_file_filter[pd_file_filter['刀盘回转速度'] > 1e-6] # 刀盘回转速度 > 0
    # pd_file_filter = pd_file_filter[pd_file_filter['刀盘马达总扭距'] > 1e-6] # 刀盘马达总扭距 > 0
    
    
    # 各列的统计信息
    data_stat = report_percentile.dataframe_percentile(pd_file_filter, col_use).T
    # 归一化
    min_col = dict()
    max_col = dict()
    med_col = dict()
    for col in col_use:
        #避免极差过大，进一步进行过滤操作
        if (data_stat.at[col, '分位点-5'] - data_stat.at[col, '分位点-1']) < 0.04*(data_stat.at[col, '最大值'] - data_stat.at[col, '最小值']):
            min_col[col] = data_stat.at[col, '分位点-1']
        else:
            min_col[col] = data_stat.at[col, '分位点-5']
        if (data_stat.at[col, '分位点-95'] - data_stat.at[col, '分位点-99']) < 0.04*(data_stat.at[col, '最大值'] - data_stat.at[col, '最小值']):
            max_col[col] = data_stat.at[col, '分位点-99']
        else:
            max_col[col] = data_stat.at[col, '分位点-95']
        med_col[col] = data_stat.at[col, '分位点-50']
        
    for col in col_use:
        pd_file_filter.loc[:, col] = np.where(pd_file_filter[col] > max_col[col], max_col[col], pd_file_filter[col])
        pd_file_filter.loc[:, col] = np.where(pd_file_filter[col] < min_col[col], min_col[col], pd_file_filter[col])
        pd_file_filter.loc[:, col] = pd_file_filter[col].apply(lambda x: (x-med_col[col])/(max_col[col]-min_col[col]) if (max_col[col]-min_col[col])>0 else None)
    
    
    '''
        制作相互关联矩阵
    '''   
    
    # 拆分训练集和测试集
    pd_file_model = pd_file_filter.copy()
    pd_file_model.dropna(inplace = True)
    
    # 输入 pd_file_filter
    col_l = col_use
    col_r = col_use
    
    # pearson 相关性分析
    path_report = 'report/correlation.csv'
    file_report = os.path.join(path_cur,path_report)
    pd_file_model[col_use].corr().reset_index().to_csv(file_report,sep = ',',index = False)
    
    # 把dataframe转为numpy array 的 list，输入和输出分开处理
    ls_in = df_to_list(pd_file_model, col_l)
    ls_out = df_to_list(pd_file_model, col_r)
    
    # 计算神经网络需要的结构形式
    dim_in = len(col_l)
    dim_out = len(col_r)
    
    # 整合输入和输出数据，转化为matrix形式
    data_train = [[sx.reshape(dim_in,1), sy.reshape(dim_out,1)] for sx, sy in zip(ls_in, ls_out)]
    
    
    # 构建训练神经网络
    num_hide = 20
    batch_size = 8
    org = [dim_in,num_hide,dim_out]
    BP = BPNNRegression(org)
    
    # 这里采用 MSGD(multi-batch stochastic gradient descending) 进行模型的训练
    ls_error_INTER = BP.MSGD(data_train, 3000, batch_size, 0.1)
    
    # 训练过程绘图
    path_fig = 'pic/interaction_process.png'
    file_fig = os.path.join(path_cur,path_fig)
    fig = plt.figure(num=0, figsize = (6, 4), dpi= 80)
    plt.plot(ls_error_INTER,'r--', label="RMSE")
    plt.legend(loc='upper left')
    plt.savefig(file_fig)
    plt.close()
    
    # 得出相互关联矩阵
    # BP中weight的每一行连接着后一层的一个神经元; 各列通过权值为前一层输入的线性变换得到本层输出
    li = len(col_l)
    lo = len(col_r)
    len_org = len(BP.sizes)-1
    MATRIX_INTERACTION = []
    
    for ci in range(li):
        MATRIX_INTERACTION.append([]) # "作用"
        for co in range(lo):
            INTER_IO = 0
            INTER_IO += pd.DataFrame(BP.weights[0])[ci].sum() # ??? 修改公式bug, 应为第一层的第ci列
            for layer in BP.weights[1:len_org-1]:
                INTER_IO += sum(sum(layer))
            INTER_IO += sum(BP.weights[-1][co])
            MATRIX_INTERACTION[-1].append(INTER_IO)
    
    DF_INTERACTION = pd.DataFrame(MATRIX_INTERACTION, index = col_l, columns = col_r)
    v_max = max(abs(DF_INTERACTION).max())
    DF_INTERACTION = DF_INTERACTION/v_max
    for col in col_use:
        DF_INTERACTION.at[col,col] = 1
    DF_INTERACTION.to_csv('report/相互关联.csv', index = True)
    
    
    #**************************** 预测代码（开始） **************************** 
    
    
    '''
        做输出值预测
    '''   
    
    
    # 训练样本集比例
    fraction_train = 0.7
    
    # 结果汇总
    dict_eval = []
    
    # 隐层节点列表
    ls_num_hide = [30]
    
    # 批次大小列表
    ls_batch_size = [14]
    
    
    # 拆分训练集和测试集
    for col in col_out:
        col_pre = col+'_pre'
        pd_file_model.loc[:,col_pre] = pd_file_model[col].shift(1)
        col_in.append(col_pre)
    pd_file_model.dropna(inplace = True)
    num_records = pd_file_model.shape[0]
    num_train = 100
    df_train = pd_file_model.iloc[:num_train,:]
    df_test = pd_file_model.loc[~pd_file_model.index.isin(df_train.index)]
    
    # 把dataframe转为numpy array 的 list，输入和输出分开处理
    ls_in = df_to_list(df_train, col_in)
    ls_out = df_to_list(df_train, col_out)
    
    # 计算神经网络需要的结构形式
    dim_in = len(col_in)
    dim_out = len(col_out)
    
    # 整合输入和输出数据，转化为matrix形式
    data_train = [[sx.reshape(dim_in,1), sy.reshape(dim_out,1)] for sx, sy in zip(ls_in, ls_out)]
    
    
    # 测量结果列表
    min_meansure = 10000000
    
    
    #**************************** Grid-Search代码（开始） **************************** 
    
    for num_hide in ls_num_hide:
        for batch_size in ls_batch_size:
    
            # 构建训练神经网络
            org = [dim_in,num_hide,num_hide,dim_out]
            BP = BPNNRegression(org)
    
            # 这里采用 MSGD(multi-batch stochastic gradient descending) 进行模型的训练
            ls_error_INTER = BP.MSGD(data_train, 1000, batch_size, 0.1)
            
    
            # 模型测试部分
            
            # 把dataframe转为numpy array 的 list，输入和输出分开处理
            ls_in = df_to_list(df_test, col_in)
            ls_out = df_to_list(df_test, col_out)
    
            
            # 因为要进行预测，所以对输入和输出值不进行合并
            data_test_x = [sx.reshape(org[0],1) for sx in ls_in]
            data_test_y = [sx.reshape(org[-1],1) for sx in ls_out]
            
            # 神经网络的预测
            pred = BP.predict(data_test_x)
            
            # 预测值和目标值的逆归一化
            yobj_an = np.array(anti_normalize(data_test_y, col_out, max_col, min_col, med_col))
            pred_an = np.array(anti_normalize(pred, col_out, max_col, min_col, med_col))
            
            # 测量结果
            measure = RMSE_M(pred_an,yobj_an)
            dict_eval.append([num_hide, batch_size, measure[0], measure[1]])
            
            if measure[0] < min_meansure:
                # 保存模型结果
                min_meansure = measure[0]
                path = 'model/final.json'
                BP.download(path)
    
    #**************************** Grid-Search代码（结束） **************************** 
    
    
    #************************ 在最优结构的基础上进行深化参数训练
    
    
    # 对神经网络进行参数优化
    path = 'model/final.json'
    BP = BPNNRegression()
    BP.upload(path)
    
    # 这里采用 MSGD(multi-batch stochastic gradient descending) 进行模型的训练
    list_error = BP.MSGD(data_train, 500, BP.mini_batch_size, BP.eta)
    
    # 把dataframe转为numpy array 的 list，输入和输出分开处理
    ls_in = df_to_list(df_train, col_in)
    ls_out = df_to_list(df_train, col_out)
    
    # 因为要进行预测，所以对输入和输出值不进行合并
    data_train_x = [sx.reshape(dim_in,1) for sx in ls_in]
    data_train_y = [sx.reshape(dim_out,1) for sx in ls_out]
    
    # 神经网络的预测
    pred = BP.predict(data_train_x)
    
    # 预测值和目标值的逆归一化
    yobj_an = np.array(anti_normalize(data_train_y, col_out, max_col, min_col, med_col))
    pred_an = np.array(anti_normalize(pred, col_out, max_col, min_col, med_col))
    
    df_yobj_an = pd.DataFrame(yobj_an)
    df_pred_an = pd.DataFrame(pred_an)
    
    # 绘图
    for c in range(dim_out):
        path_fig = 'pic/final_train'+ col_out[c] +'.png'
        file_fig = os.path.join(path_cur,path_fig)
        fig = plt.figure(num=c, figsize = (6, 4), dpi= 80)
        plt.plot(df_yobj_an[c].tolist(),'r-', label="目标值")
        plt.plot(df_pred_an[c].tolist(),'b--', label="预测值")
        plt.legend(loc='upper left')
        plt.title(col_out[c])
        plt.savefig(file_fig)
        plt.close()
    
        # 训练过程绘图
        path_fig = 'pic/'+col_out[c]+ 'final_train_process.png'
        file_fig = os.path.join(path_cur,path_fig)
        fig = plt.figure(num=0, figsize = (6, 4), dpi= 80)
        plt.plot(list_error,'r--', label="RMSE")
        plt.legend(loc='upper left')
        plt.title(col_out[c] + '_训练过程')
        plt.savefig(file_fig)
        plt.close()
        
    # 输出数据
    df_pred_an.columns = col_out
    df_yobj_an.columns = col_out
    df_pred_an.to_csv('model/模型训练预测结果.csv',index=False)
    df_yobj_an.to_csv('model/模型训练目标输出.csv',index=False)
    
    #************************ 使用最新的50个样本点进行模型测试，观察最终效果
    
    # 把dataframe转为 numpy array 的 list，输入和输出分开处理
    ls_in = df_to_list(df_test, col_in)
    ls_out = df_to_list(df_test, col_out)
    
    
    # 因为要进行预测，所以对输入和输出值不进行合并
    data_test_x = [sx.reshape(dim_in,1) for sx in ls_in]
    data_test_y = [sx.reshape(dim_out,1) for sx in ls_out]
    
    # 神经网络的预测
    pred = BP.predict(data_test_x)
    
    # 预测值和目标值的逆归一化
    yobj_an = np.array(anti_normalize(data_test_y, col_out, max_col, min_col, med_col))
    pred_an = np.array(anti_normalize(pred, col_out, max_col, min_col, med_col))
    
    df_yobj_an = pd.DataFrame(yobj_an)
    df_pred_an = pd.DataFrame(pred_an)
    df_pred_an.columns = col_out
    df_yobj_an.columns = col_out
    df_pred_an.to_csv('model/模型预测结果.csv',index=False)
    df_yobj_an.to_csv('model/模型目标输出.csv',index=False)
    
    # 绘图
    for c in range(dim_out):
        cc = col_out[c]
        path_fig = 'pic/final_test'+ col_out[c] +'.png'
        file_fig = os.path.join(path_cur,path_fig)
        fig = plt.figure(num=c, figsize = (6, 4), dpi= 80)
        plt.plot(df_yobj_an[cc].tolist(),'r-', label="目标值")
        plt.plot(df_pred_an[cc].tolist(),'b--', label="预测值")
        plt.legend(loc='upper left')
        RMSE = RMSE_M(df_pred_an[cc].tolist(), df_yobj_an[cc].tolist())[1]
        plt.title(col_out[c] + '->RSE: ' +str(RMSE))
        plt.savefig(file_fig)
        plt.close()
    
    #**************************** 预测代码（结束） **************************** 
    
