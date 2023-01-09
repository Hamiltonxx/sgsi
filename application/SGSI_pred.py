from flask import request, send_file
from application import app
import os
import json
from application.bp_final import BPNNRegression, df_to_list, anti_normalize, RMSE_M
from application.report_percentile import dataframe_percentile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
#matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']= False    # 正常显示负号
fpath = '/home/dlserver/projects/SGSI_Pred/simfang.ttf'
prop = fm.FontProperties(fname=fpath)
import pandas as pd
import numpy as np
from config import r, PRJROOT
encodings = ['gb18030', 'utf-8']

# @app.route("/helloworld_test", methods=["GET"])
# def helloworld_test():
#     return "Hello World"

# curl -X POST -d '{"project_name":"aaa","task":"train"}'  https://dev.yijianar.com:8441/progress
@app.route("/progress", methods=["POST"])
def progress():
    args = request.get_json(force=True)
    project_name, task = args['project_name'], args['task']
    return r.get(f'{project_name}_{task}')

# curl -X POST -F filename=@312_1.csv -F project_name=aaa https://dev.yijianar.com:8441/Upload_Train
@app.route("/Upload_Train", methods=["POST"])
def uploadTrain():
    file = request.files['filename']
    for encoding in encodings:
        try:
            pd_file = pd.read_csv(file, encoding=encoding)
            #print(file)
            break
        except:
            continue
    else:
        return '请上传utf-8格式 或 gb18030格式的csv文件'
    
    if '管片号码' not in pd_file.columns:
        return 'csv文件需包含 管片号码 列'
    if '舱内土压' not in pd_file.columns:
        return 'csv文件需包含 舱内土压 列'
    if '刀盘旋转速度' not in pd_file.columns:
        return 'csv文件需包含 刀盘旋转速度 列'
    if '刀盘扭矩' not in pd_file.columns:
        return 'csv文件需包含 刀盘扭矩 列'
    if '总推力' not in pd_file.columns:
        return 'csv文件需包含 总推力 列'
    if '推进速度' not in pd_file.columns:
        return 'csv文件需包含 推进速度 列'
    if '地表变形' not in pd_file.columns:
        return 'csv文件需包含 地表变形 列'
    print('-----------------')
    project_name = request.form.get('project_name')
    col_use = ['舱内土压', '刀盘旋转速度', '刀盘扭矩', '总推力', '推进速度', '地表变形']
    pd_file_filter = pd_file.copy()[col_use]

    data_stat = dataframe_percentile(pd_file_filter, col_use).T
    min_col = dict()
    max_col = dict()
    med_col = dict()
    for col in col_use:
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
    
    pd_file_model = pd_file_filter.copy()
    pd_file_model.dropna(inplace=True)
    col_in = ['舱内土压', '刀盘旋转速度','推进速度']
    col_out = ['总推力', '刀盘扭矩', '地表变形']
    # 训练样本集比例
    fraction_train = 0.7
    # 隐层节点列表
    ls_num_hide = [30, 30]
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
            model6x3 = BPNNRegression(org)
    
            # 这里采用 MSGD(multi-batch stochastic gradient descending) 进行模型的训练
            ls_error_INTER = model6x3.MSGD(data_train, 3000, batch_size, 0.1, task = 'train', project_name = project_name)
            plt.plot(ls_error_INTER)
            plt.xlabel('训练迭代次数',fontproperties = prop)
            plt.ylabel('RMSE损失', fontproperties = prop)
            plt.title(f'地层参数模型训练损失函数变化图', fontproperties = prop)
            plt.savefig(f'./pic/{project_name}_modelTrain.jpg')
            plt.close()
    
            # 模型测试部分
            
            # 把dataframe转为numpy array 的 list，输入和输出分开处理
            ls_in = df_to_list(df_test, col_in)
            ls_out = df_to_list(df_test, col_out)
    
            
            # 因为要进行预测，所以对输入和输出值不进行合并
            data_test_x = [sx.reshape(org[0],1) for sx in ls_in]
            data_test_y = [sx.reshape(org[-1],1) for sx in ls_out]
            
            # 神经网络的预测
            pred = model6x3.predict(data_test_x)
            
            # 预测值和目标值的逆归一化
            yobj_an = np.array(anti_normalize(data_test_y, col_out, max_col, min_col, med_col))
            pred_an = np.array(anti_normalize(pred, col_out, max_col, min_col, med_col))
            
            # 测量结果
            measure = RMSE_M(pred_an,yobj_an)
            
            if measure[0] < min_meansure:
                # 保存模型结果
                min_meansure = measure[0]
                path = f'./data/model/{project_name}6x3.json'
                model6x3.download(path)
    return f'./pic/{project_name}_modelTrain.jpg'

@app.route("/file", methods=["POST"])
def downloadFile():
    args = request.get_json(force=True)
    filepath = args["filepath"]
    return send_file(f"{PRJROOT}/{filepath}")


# curl -X POST -F filename=@pingjia1.csv -F project_name=aaa https://dev.yijianar.com:8441/SGSI_Evaluate
@app.route("/SGSI_Evaluate", methods=["POST"])
def evaluateSGSI():
    file = request.files['filename']
    for encoding in encodings:
        try:
            pd_file = pd.read_csv(file, encoding=encoding)
            break
        except:
            continue
    else:
        return '请上传utf-8格式 或 gb18030格式的csv文件'
    if '管片号码' not in pd_file.columns:
        return 'csv文件需包含 管片号码 列'
    if '推力指数' not in pd_file.columns:
        return 'csv文件需包含 推力指数 列'
    if '扭矩指数' not in pd_file.columns:
        return 'csv文件需包含 扭矩指数 列'
    if '推进速度' not in pd_file.columns:
        return 'csv文件需包含 推进速度 列'
    if '刀盘旋转速度' not in pd_file.columns:
        return 'csv文件需包含 刀盘旋转速度 列'
    if '能耗' not in pd_file.columns:
        return 'csv文件需包含 能耗 列'
    if '地表变形' not in pd_file.columns:
        return 'csv文件需包含 地表变形 列'
    project_name = request.form.get('project_name')
    col_use = ['推力指数', '扭矩指数', '推进速度', '刀盘旋转速度', '能耗', '地表变形']
    pd_file_filter = pd_file.copy()[col_use]
    pd_file_filter.to_csv(f'./data/csv_6/{project_name}.csv')
    data_stat = dataframe_percentile(pd_file_filter, col_use).T
    min_col = dict()
    max_col = dict()
    med_col = dict()
    for col in col_use:
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
    pd_file_model = pd_file_filter.copy()
    pd_file_model.dropna(inplace = True)
    col_l = col_use
    col_r = col_use
    ls_in = df_to_list(pd_file_model, col_l)
    ls_out = df_to_list(pd_file_model, col_r)
    dim_in = len(col_l)
    dim_out = len(col_r)
    data_train = [[sx.reshape(dim_in,1), sy.reshape(dim_out,1)] for sx, sy in zip(ls_in, ls_out)]
    num_hide = 20
    batch_size = 8
    org = [dim_in,num_hide,dim_out]
    model6x6 = BPNNRegression(org)
    ls_error_INTER = model6x6.MSGD(data_train, 3000, batch_size, 0.1, task='evaluate', project_name = project_name)
    plt.plot(ls_error_INTER)
    plt.xlabel('训练迭代次数',fontproperties = prop)
    plt.ylabel('RMSE损失', fontproperties = prop)
    plt.title(f'相互作用矩阵BP神经网络训练损失函数变化图', fontproperties = prop)
    plt.savefig(f'./pic/{project_name}_evaluateModelTrain.jpg')
    plt.close()
    li = len(col_l)
    lo = len(col_r)
    len_org = len(model6x6.sizes)-1
    MATRIX_INTERACTION = []
    for ci in range(li):
        MATRIX_INTERACTION.append([]) # "作用"
        for co in range(lo):
            INTER_IO = 0
            INTER_IO += pd.DataFrame(model6x6.weights[0])[ci].sum() # ??? 修改公式bug, 应为第一层的第ci列
            for layer in model6x6.weights[1:len_org-1]:
                INTER_IO += sum(sum(layer))
            INTER_IO += sum(model6x6.weights[-1][co])
            MATRIX_INTERACTION[-1].append(INTER_IO)
    DF_INTERACTION = pd.DataFrame(MATRIX_INTERACTION, index = col_l, columns = col_r)
    v_max = DF_INTERACTION.values.max()
    v_min = DF_INTERACTION.values.min()
    DF_INTERACTION = (DF_INTERACTION - v_min)/ (v_max - v_min)
    for col in col_use:
        DF_INTERACTION.at[col,col] = 1
    MATRIX_INTERACTION = DF_INTERACTION.values
    print(MATRIX_INTERACTION)
    interaction_list = list(MATRIX_INTERACTION.reshape(1, -1)[0])
    interaction_string = ""
    for item in interaction_list:
        interaction_string += str(item)
        interaction_string +=' '
    Ci = np.sum(MATRIX_INTERACTION, axis = 0)
    Ei = np.sum(MATRIX_INTERACTION, axis = 1)
    print(Ci, Ei)
    CE_sum = np.sum(Ci) + np.sum(Ei)
    w = [(Ci[i] + Ei[i]) / CE_sum for i in range(len(Ci))]
    pd_file_copy = pd_file.copy()[col_use]
    pd_file_copy = get_rank(pd_file_copy, col_use)
    rank_matrix = pd_file_copy.values
    SGSI = []
    for i in range(len(pd_file_copy)):
        tmp  = [w[j] * rank_matrix[i][j] for j in range(len(w))]
        SGSI.append(sum(tmp) * 100)
    ringNo = pd_file['管片号码'].values
    print(ringNo)
    print(SGSI)
    plt.plot(ringNo.reshape(1,-1), np.array(SGSI).reshape(1,-1), 'ro-', marker = 'o', markerfacecolor='white')
    plt.legend(['SGSI评价曲线'], prop=prop)
    plt.title(f'{project_name} SGSI评价', fontproperties = prop)
    plt.xlabel('环号', fontproperties = prop)
    plt.ylabel('SGSI评分', fontproperties = prop)
    plt.grid(color = 'gray', linestyle = '--')
    plt.savefig(f'./pic/{project_name}_SGSIEvaluate.jpg', dpi = 1000)
    plt.close()
    return json.dumps({"evaluateModelTrainPic":f'./pic/{project_name}_evaluateModelTrain.jpg',"interaction": interaction_string, "SGSIEvaluatePic":f'./pic/{project_name}_SGSIEvaluate.jpg'})

# curl -F filename=@312_1.csv -F project_name=aaa -F Do=6 -F Di=5.25 -F Dc=6.14 https://dev.yijianar.com:8441/SGSI_Prediction
@app.route("/SGSI_Prediction", methods=["POST"])
def predictSGSI():
    project_name = request.form.get('project_name')
    Do, Di, Dc = float(request.form.get('Do')), float(request.form.get('Di')), float(request.form.get('Dc'))
    models = os.listdir('./data/model/')
    model6x3 = BPNNRegression()
    for model in models:
        if project_name in model:
            model6x3.upload('./data/model/' + model)
            break
    else:
        return '请先训练用于预测下一环 \'总推力\' \'刀盘扭矩\' \'地表变形\' 的模型'
    file = request.files['filename']
    for encoding in encodings:
        try:
            pd_file = pd.read_csv(file, encoding=encoding)
            break
        except:
            continue
    else:
        return '请上传utf-8格式 或 gb18030格式的csv文件'

    if '管片号码' not in pd_file.columns:
        return 'csv文件需包含 管片号码 列'
    if '舱内土压' not in pd_file.columns:
        return 'csv文件需包含 舱内土压 列'
    if '刀盘旋转速度' not in pd_file.columns:
        return 'csv文件需包含 刀盘旋转速度 列'
    if '刀盘扭矩' not in pd_file.columns:
        return 'csv文件需包含 刀盘扭矩 列'
    if '总推力' not in pd_file.columns:
        return 'csv文件需包含 总推力 列'
    if '推进速度' not in pd_file.columns:
        return 'csv文件需包含 推进速度 列'
    if '地表变形' not in pd_file.columns:
        return 'csv文件需包含 地表变形 列'

    # 预测下一环三参数
    col_use = ['舱内土压', '刀盘旋转速度', '刀盘扭矩', '总推力', ' 推进速度', '地表变形']
    pd_file_filter = pd_file.copy()[col_use]
    data_stat = dataframe_percentile(pd_file_filter, col_use).T
    min_col = dict()
    max_col = dict()
    med_col = dict()
    for col in col_use:
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
    col_in = col_use
    col_out = ['总推力', '刀盘扭矩', '地表变形']
    ls_in = df_to_list(pd_file_filter, col_in)
    dim_in = len(col_in)
    data_train_x = [sx.reshape(dim_in,1) for sx in ls_in]
    pred = model6x3.predict(data_train_x)
    pred_an = np.array(anti_normalize(pred, col_out, max_col, min_col, med_col))
    pred_df = pd.DataFrame(pred_an)
    pred_df.columns = ['总推力', '刀盘扭矩', '地表变形']
    pred_df['舱内土压'] = pd_file.copy()['舱内土压']
    pred_df['刀盘旋转速度'] = pd_file.copy()['刀盘旋转速度']
    pred_df['推进速度'] = pd_file.copy()['推进速度']
    pred_df = get_six_parameters(pred_df, Do, Di, Dc, col_use=['推力指数', '扭矩指数', '推进速度', '刀盘旋转速度', '能耗', '地表变形'])
    pred_df['管片号码'] = pd_file['管片号码']
    pred_df.dropna(inplace=True)
    # 预测下一环SGSI
    col_use=['推力指数', '扭矩指数', '推进速度', '刀盘旋转速度', '能耗', '地表变形']
    pd_file_filter = pred_df.copy()[col_use]
    data_stat = dataframe_percentile(pd_file_filter, col_use).T
    min_col = dict()
    max_col = dict()
    med_col = dict()
    for col in col_use:
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
    col_l = col_use
    col_r = col_use
    ls_in = df_to_list(pd_file_filter, col_l)
    ls_out = df_to_list(pd_file_filter, col_r)
    dim_in = len(col_l)
    dim_out = len(col_r)
    data_train = [[sx.reshape(dim_in,1), sy.reshape(dim_out,1)] for sx, sy in zip(ls_in, ls_out)]
    num_hide = 20
    batch_size = 8
    org = [dim_in,num_hide,dim_out]
    model6x6 = BPNNRegression(org)
    ls_error_INTER = model6x6.MSGD(data_train, 3000, batch_size, 0.1, task='predict', project_name = project_name)
    li = len(col_l)
    lo = len(col_r)
    len_org = len(model6x6.sizes)-1
    MATRIX_INTERACTION = []
    for ci in range(li):
        MATRIX_INTERACTION.append([]) # "作用"
        for co in range(lo):
            INTER_IO = 0
            INTER_IO += pd.DataFrame(model6x6.weights[0])[ci].sum() # ??? 修改公式bug, 应为第一层的第ci列
            for layer in model6x6.weights[1:len_org-1]:
                INTER_IO += sum(sum(layer))
            INTER_IO += sum(model6x6.weights[-1][co])
            MATRIX_INTERACTION[-1].append(INTER_IO)
    DF_INTERACTION = pd.DataFrame(MATRIX_INTERACTION, index = col_l, columns = col_r)
    v_max = DF_INTERACTION.values.max()
    v_min = DF_INTERACTION.values.min()
    DF_INTERACTION = (DF_INTERACTION - v_min)/ (v_max - v_min)
    for col in col_use:
        DF_INTERACTION.at[col,col] = 1
    MATRIX_INTERACTION = DF_INTERACTION.values
    Ci = np.sum(MATRIX_INTERACTION, axis = 0)
    Ei = np.sum(MATRIX_INTERACTION, axis = 1)
    CE_sum = np.sum(Ci) + np.sum(Ei)
    w = [(Ci[i] + Ei[i]) / CE_sum for i in range(len(Ci))]
    pred_df_copy = pred_df.copy()[col_use]
    pred_df_copy = get_rank(pred_df_copy, col_use)
    rank_matrix = pred_df_copy.values
    SGSI_pred = []
    for i in range(len(pred_df_copy)):
        tmp  = [w[j] * rank_matrix[i][j] for j in range(len(w))]
        SGSI_pred.append(sum(tmp) * 100)
    ringNo_pred = pred_df['管片号码'].values

    # 求SGSI真实值
    col_use = ['管片号码','舱内土压', '刀盘旋转速度', '刀盘扭矩', '总推力', '推进速度', '地表变形']
    real_df = pd_file.copy()[col_use]
    col_out = ['总推力', '刀盘扭矩', '地表变形']
    for col in col_out:
        real_df[col] = real_df[col].shift(-1)
    real_df.dropna(inplace=True)
    ringNo_real = real_df['管片号码'].values
    col_use=['推力指数', '扭矩指数', '推进速度', '刀盘旋转速度', '能耗', '地表变形']
    real_df = get_six_parameters(real_df, Do, Di, Dc, col_use=['推力指数', '扭矩指数', '推进速度', '刀盘旋转速度', '能耗', '地表变形'])
    real_df_copy = real_df.copy()[col_use]
    real_df_copy = get_rank(real_df_copy, col_use)
    rank_matrix = real_df_copy.values
    SGSI_real = []
    for i in range(len(real_df_copy)):
        tmp  = [w[j] * rank_matrix[i][j] for j in range(len(w))]
        SGSI_real.append(sum(tmp) * 100)
    
    plt.plot(ringNo_pred.reshape(1,-1), np.array(SGSI_pred).reshape(1,-1), 'ro-', markerfacecolor='white', label='SGSI预测值')
    plt.plot(ringNo_real.reshape(1,-1), np.array(SGSI_real).reshape(1,-1), 'go-', markerfacecolor='white', label='SGSI真实值')
    plt.title(f'SGSI实时评估与动态预测曲线图', fontproperties = prop)
    plt.legend([f'{project_name} SGSI真实值'], prop = prop)
    plt.xlabel('环号', fontproperties = prop)
    plt.ylabel('适应性指数(SGSI)', fontproperties = prop)
    plt.grid(color = 'gray', linestyle = '--')
    plt.savefig(f'./pic/{project_name}_SGSIPrediction.jpg', dpi = 1000)
    plt.close()
    return f'pic/{project_name}_SGSIPrediction.jpg'
    
def get_six_parameters(pred_df, Do, Di, Dc, col_use):
    for i in range(len(pred_df)):
        pred_df.loc[i, '推力指数'] = calc_tlzs(pred_df.loc[i, '总推力'], Do, Di)
        pred_df.loc[i, '扭矩指数'] = calc_njzs(pred_df.loc[i, '刀盘扭矩'], Dc)
        pred_df.loc[i, '贯入度'] = calc_grd(pred_df.loc[i, '推进速度'], pred_df.loc[i, '刀盘旋转速度'])
        pred_df.loc[i, '能耗'] = calc_nh(pred_df.loc[i, '总推力'],pred_df.loc[i, '贯入度'], pred_df.loc[i, '刀盘扭矩'], Dc)
    pred_df = pred_df[col_use]
    return pred_df

def calc_tlzs(F, Do, Di):
    return 4 * F / np.pi / (Do ** 2 - Di ** 2)

def calc_njzs(T, Dc):
    return T / (Dc ** 3)

def calc_grd(PR, Rc):
    if Rc == 0 and PR == 0 :
        return 0
    if Rc == 0:
        return 1e8
    return PR / Rc

def calc_nh(F, P, T, Dc):
    if P == 0:
        return 4 * F / (np.pi * Dc **2)
    return 4 * (F * P + 2 * np.pi * T) / (np.pi * Dc**2 * P)

def get_rank(pd_file_copy, col_use):
    for i in range(len(pd_file_copy)):
        pd_file_copy.loc[i, '推力指数'] = tlzs_rank(pd_file_copy.loc[i, '推力指数'])
        pd_file_copy.loc[i, '扭矩指数'] = njzs_rank(pd_file_copy.loc[i, '扭矩指数'])
        pd_file_copy.loc[i, '推进速度'] = tjsd_rank(pd_file_copy.loc[i, '推进速度'])
        pd_file_copy.loc[i, '刀盘旋转速度'] = dpxzsd_rank(pd_file_copy.loc[i, '刀盘旋转速度'])
        pd_file_copy.loc[i, '能耗'] = nh_rank(pd_file_copy.loc[i, '能耗'])
        pd_file_copy.loc[i, '地表变形'] = dbbx_rank(pd_file_copy.loc[i, '地表变形'])
    return pd_file_copy

def tlzs_rank(tlzs):
    border = [4500 - 300 * i for i in range(10)]
    i = 0
    for item in border:
        if tlzs >= item:
            return i * 0.1
        i += 1
    return i * 0.1

def njzs_rank(njzs):
    border = [23 - 2 * i for i in range(10)]
    i = 0
    for item in border:
        if njzs >= item:
            return i * 0.1
        i += 1
    return i * 0.1

def tjsd_rank(tjsd):
    border = [5 + 5 * i for i in range(10)]
    i = 0
    for item in border:
        if tjsd < item:
            return i * 0.1
        i += 1
    return i * 0.1

def dpxzsd_rank(dpxzsd):
    border = [0.3 + 0.3 * i for i in range(10)]
    i = 0
    for item in border:
        if dpxzsd < item:
            return i * 0.1
        i += 1
    return i * 0.1

def nh_rank(nh):
    border = [35000 - 3000 * i for i in range(10)]
    i = 0
    for item in border:
        if nh >= item:
            return i * 0.1
        i += 1
    return i * 0.1

def dbbx_rank(dbbx):
    border = [50 - 5 * i for i in range(10)]
    i = 0
    for item in border:
        if dbbx >= item:
            return i * 0.1
        i += 1
    return i * 0.1
