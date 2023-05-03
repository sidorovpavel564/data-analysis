import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


data_dict = {
    'col1': [-0.18, 0.67, 1.61, 1.44, -1.6, 1.17, 1.09, 0.25, 0.54, 0.08, 1.03, 0.45, -0.01, -0.17, 2.34, -0.13, 1.44, -1.62, -1.96, -0.98, 1.86, 0.55, 0.66, 0.73, 1.06, -1.08, 0.43, -0.36, -0.33, -0.76, 0.2, -0.82, -0.74, -1.13, -0.6, -1.3, -0.33, 0.33, -0.11, 1.1, 0.26, -1.3, 0.01, 0.03, 0, -0.24, -1.25, -0.87, -1.26, 1.21, 1.43, 0.89, 1.25, 1.1, 2.04, 0.14, 0.97, 0.02, -0.03, 1.27, 0.12, 0.25, 0.05, 1.19, -0.5, 0.97, 0.64, 1.24, 1.04, 0.1, -0.96, 2.01, 0.85, -1.71, 0.44, 0.09, 0.09, -1.32, -0.86, 0.98, 3.19, -0.02, 0.65, -0.1, 0.24, -0.53, -0.32, 0.18, -0.62, 2.05, -1.13, -0.6, -0.27, 0, 0.38, 0.48, 0.35, -0.44, -0.29, -0.16],
    'col2': [-0.65, 0.3, -0.95, 0.83, -1.02, -0.8, 0.53, -0.4, -0.02, -0.69, 1.6, -0.45, -1.04, -0.7, 0.07, -1.55, 0.43, 0.35, -0.01, -1.06, -0.64, 1.63, -1.1, -0.53, -0.98, 0.92, 2.26, -0.76, 0.25, -0.11, -0.75, 1.11, 1.93, -1.29, 1.69, -0.12, 0.87, -1.54, 0.44, -0.28, 1.5, 0.11, -0.68, -0.54, -1.41, -1.91, -0.63, -1.78, 0.59, 0.63, 0.16, -1.04, -2.35, 0.52, -0.99, 0.02, -0.45, -1.71, -0.03, 0.8, 1.95, -1.17, -0.6, -0.11, -0.11, -1.39, -0.55, 0.64, -0.05, 0.06, 0.06, 0.66, -1.11, -0.99, 0.42, -0.85, 0.63, -0.81, -0.85, 0.28, -0.26, -0.82, -0.35, -0.82, 1.4, 0.02, -1.13, 0.83, -1.34, -0.73, -1.43, -1.13, 1.48, -0.04, 0.18, -0.85, -0.04, -1.12, 1.08, -1.66],
    'col3': ['1.17', 0.12, 0.6, 0.2, 0.44, 1.21, 0.2, -0.98, -0.45, 1.4, 0.27, -0.2, 0.29, 1.42, 1.37, -0.82, 1.67, -0.05, -0.45, -0.29, 1.06, 0.04, 1.54, 0.35, -1.44, 0.29, -2.08, -1.71, 0.85, -0.79, -0.95, -0.23, 0.27, 0.1, -0.65, 0.2, -0.94, 0.09, -2.18, 0.79, 0.33, -0.29, 0.3, -0.28, -1.88, -0.35, -1.56, 0.75, 1.79, -0.93, 0.88, 0.31, -0.8, 0.48, 0.06, -0.31, -1.15, -0.69, -1.94, -0.73, -0.5, -1.82, 1.16, -1.2, -0.16, -0.72, -2.57, -0.69, -0.72, 1.04, -0.44, -1.39, -1.1, -0.49, 0.96, 1.02, -0.28, -0.28, 0.78, 1.62, 0.31, -2.01, 0.31, -2.27, 0.47, 0.33, -0.91, 0.12, 1.25, 3.13, 0.73, -0.76, -0.35, 0.65, -0.23, 1.4, 1.92, 0.79, -1.98, 0.03],
}
data = pd.DataFrame(data=data_dict)
# data.hist(bins=30, figsize=(30, 30), grid=False)
# plt.savefig('FIGURE')

# sns.histplot(data=data['col1'])
# plt.savefig('SNS')

# print(stats.normaltest(data['col1']))
# print(stats.normaltest(data['col2']))
# print(stats.normaltest(data['col3']))

# sns.histplot(data=data['col1'])
# plt.savefig('SNScol1')

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


def check_dataframe_types(df: pd.DataFrame) -> list:
    """
    Receives pandas.DataFrame and forms list of columns that are not int or 
    float type.
    :param pd.DataFrame df: Dataset in pd.DataFrame format.
    :return: List of columns that are not int or float type.
    """
    allowed_types = [int, float]
    invalid_columns = []

    for column in df.columns:
        column_type = df[column].dtype
        if column_type not in allowed_types:
            invalid_columns.append(column)
    return invalid_columns

print(check_dataframe_types(data))
