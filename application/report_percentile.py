import pandas as pd
import numpy as np



def profile(df, ls_col):
    # 数据描述
    ls_col_df = df.columns.tolist()
    assert ((len(ls_col_df) > 0) and (set(ls_col).issubset(set(ls_col_df)))), '所选列不在数据表中，请重新检查'
    data_describe = {}
    dimension = ['总数','非空数','最小值','分位点-1','分位点-5','分位点-10','分位点-50','平均值',
                 '分位点-90','分位点-95','分位点-99','最大值']
    for col in ls_col:
        size = df[col].shape[0]
        count = df[col].count()
        min_col = df[col].min()
        percentile_1 = df[col].quantile(0.01)
        percentile_5 = df[col].quantile(0.05)
        percentile_10 = df[col].quantile(0.1)
        percentile_50 = df[col].quantile(0.5)
        mean_col = df[col].mean()
        percentile_90 = df[col].quantile(0.9)
        percentile_95 = df[col].quantile(0.95)
        percentile_99 = df[col].quantile(0.99)
        max_col = df[col].max()
        data_describe[col] = [size, count, min_col, percentile_1, percentile_5, percentile_10, percentile_50, mean_col, 
                              percentile_90, percentile_95, percentile_99, max_col]
                              
    pd_describe = pd.DataFrame.from_dict(data_describe, orient='index',columns=dimension).T
    return pd_describe
    
    
def dataframe_percentile(df, ls_col):
    # 数据描述
    ls_col_df = df.columns.tolist()
    assert ((len(ls_col_df) > 0) and (set(ls_col).issubset(set(ls_col_df)))), '所选列不在数据表中，请重新检查'
    data_describe = {}
    ls_percentile = [i for i in range(101)]
    dimension = ['总数','非空数','最小值','平均值','最大值'] + ['分位点-' + str(i) for i in ls_percentile]
    for col in ls_col:
        value = []
        value.append(df[col].shape[0]) # 总数
        value.append(df[col].count()) # 非空数
        value.append(df[col].min()) # 最小值
        value.append(df[col].mean()) # 平均值
        value.append(df[col].max()) # 最大值
        for i in ls_percentile:
            value.append(df[col].quantile(i/100.0))
        data_describe[col] = value

    pd_describe = pd.DataFrame.from_dict(data_describe, orient='index',columns=dimension).T
    return pd_describe


def list_percentile(df,ls_col):
    # 数据描述
    ls_col_df = df.columns.tolist()
    assert ((len(ls_col_df) > 0) and (set(ls_col).issubset(set(ls_col_df)))), '所选列不在数据表中，请重新检查'
    data_describe = {}
    ls_percentile = [i for i in range(101)]
    dimension = ['总数','非空数','最小值','平均值','最大值'] + ['分位点-' + str(i) for i in ls_percentile]
    for col in ls_col:
        value = []
        value.append(df[col].shape[0]) # 总数
        value.append(df[col].count()) # 非空数
        value.append(df[col].min()) # 最小值
        value.append(df[col].mean()) # 平均值
        value.append(df[col].max()) # 最大值
        for i in ls_percentile:
            value.append(df[col].quantile(i/100.0))
        data_describe[col] = value
    pd_describe = pd.DataFrame.from_dict(data_describe, orient='index',columns=dimension).T
    return pd_describe

