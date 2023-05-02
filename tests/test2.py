import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from scipy import stats
import numpy as np


data = pd.read_csv('PerfIndicators_RusUniversities_01012018/data.csv', delimiter=';')
data.drop(['federal_district', 'federal_district_short', 'region_code',
       'region_name', 'okato', 'id', 'name', 'name_short', 'year'], axis=1, inplace=True)
data.dropna(inplace=True)

# sns.histplot(data=data['e8'])
# plt.savefig('SNS2')

# data.hist(column='e8', bins=250)
# plt.savefig('PD2')

# mask = data['e8'].between(0, 20)
# count = mask.sum()
# print(count)

# print(stats.normaltest(data))

# alpha = 1e-3
# threshold = 0.7
# p_values = stats.normaltest(data)[1]
# count = sum(value < alpha for value in p_values)
# # print(count / len(p_values) > threshold)
# # print(all(value < alpha for value in stats.normaltest(data)[1]))
# corr_method = 'pearson'
# if count / len(p_values) > threshold:
#     corr_method = 'kendall'
# print(corr_method)

# Calculate the IQR for each column
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3 - q1

# Set a threshold for outlier detection (e.g., IQR multiplied by 1.5)
iqr_threshold = 1.5

# Identify outliers using IQR
outliers = ((data < (q1 - iqr_threshold * iqr)) | (data > (q3 + iqr_threshold * iqr)))

# Print the outliers
outliers_columns = outliers.any()
if outliers_columns.any():
    print("Outliers (IQR):")
    for col in outliers_columns.index:
        col_outliers = data.loc[outliers[col], col]
        print(f"Outliers in column {col}:")
        print(col_outliers)
else:
    print("No outliers (IQR)")