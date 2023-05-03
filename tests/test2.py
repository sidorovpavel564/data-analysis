import numpy as np
from scipy import stats
import pandas as pd
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


data = pd.read_csv(
    'PerfIndicators_RusUniversities_01012018/data.csv', delimiter=';')
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


def has_outliers(df):
    num_columns = df.shape[1]
    num_columns_with_outliers = 0

    for column in df.columns:
        data = df[column]
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # Define outliers using the 1.5*IQR criterion
        outliers = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)

        if outliers.any():
            num_columns_with_outliers += 1

    percentage = (num_columns_with_outliers / num_columns) * 100

    return percentage, percentage > 85


# print(has_outliers(data))

data.info()
