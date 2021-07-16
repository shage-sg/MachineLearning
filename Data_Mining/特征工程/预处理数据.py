# encoding=utf-8

# Time : 2021/4/29 15:20 

# Author : 啵啵

# File : 预处理数据.py

# Software: PyCharm

from pprint import pprint
import pandas as pd
import xgboost as xgb
if __name__ == '__main__':
    fpath = r'.\data\食堂调查问卷数据.xlsx'
    # 默认使用xlrd,版本不兼容问题，我们这里使用openpyxl
    CanteenDifferenceQuestionnaire = pd.read_excel(fpath, engine='openpyxl')
    # 设置显示最大列数
    pd.set_option('display.max_columns', 100)
    # 设置显示数值的精度
    pd.set_option('precision', 3)
    # 格式化索引
    FirstData = CanteenDifferenceQuestionnaire[CanteenDifferenceQuestionnaire.columns[:15]]
    SecondData = CanteenDifferenceQuestionnaire[CanteenDifferenceQuestionnaire.columns[15:]] \
        .reindex_like(FirstData)
    # 统计：离散值
    pprint(FirstData.describe())
    # 统计占比
    pprint({column: [*zip(FirstData.loc[:, column].value_counts(normalize=True).index,
                        map(lambda data:round(data,3),
                            FirstData.loc[:, column].value_counts(normalize=True).values))]
            for column in FirstData.columns})
