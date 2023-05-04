import json
import mimetypes

import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                             calculate_kmo)
from scipy import stats


def filename_extension_is_csv(file_path: str) -> bool:
    """
    Checks that the filename extension is ".csv".
    :param str file_path: File path. Example: 'data.csv'.
    :return: True if filename extension is ".csv". False otherwise.
    """
    mimestart = mimetypes.guess_type(file_path)[0]
    if mimestart != None:
        mimestart = mimestart.split("/")[1]
        if mimestart == "csv":
            return True
    return False


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


def get_outliers_percentage(df: pd.DataFrame) -> float:
    """
    Receives pandas.DataFrame, calculates IQR and returns percentage of columns 
    with outliers.
    :param pd.DataFrame df: Dataset in pd.DataFrame format.
    :return: Percentage of columns with outliers.
    """
    num_columns = df.shape[1]
    num_columns_with_outliers = 0

    # Calculate the IQR for each column
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

    return percentage


def perform_correlation_analysis(csv_file_path: str, output_file: str, dropna: bool = True) -> None:
    """
    Creates pd.DataFrame from open csv-file, removes missing values if dropna=True, checks 
    that the data are numeric, checks data normal distribution, checks outliers, 
    defines correlation method, calculates correlation matrix and writes results 
    to json-file.
    :param str csv_file_path: Dataset filepath. The filename extension should 
    be "csv".
    :param str output_file: Result filepath. The filename extension should be 
    "json".
    :param bool dropna: Remove missing values if True. Default True.
    :returns: None
    """
    if filename_extension_is_csv(csv_file_path):
        # Read the CSV file into a Pandas DataFrame
        data = pd.read_csv(csv_file_path)

        # Remove missing values
        data.dropna(inplace=True)

        # Check variables types
        invalid_columns = check_dataframe_types(data)

        if len(invalid_columns) == 0:

            # Check data normal distribution and define corr method
            alpha = 1e-3
            threshold = 0.7
            p_values = stats.normaltest(data)[1]
            count = sum(value < alpha for value in p_values)
            corr_method = 'pearson'
            if count / len(p_values) > threshold:
                corr_method = 'kendall'

            # Check outliers
            outliers_percentage = get_outliers_percentage(data)
            if outliers_percentage > 85:
                corr_method = 'kendall'

            # Perform Bartlett's test
            chi_square_value, p_value = calculate_bartlett_sphericity(data)
            bartletts_test = {'chi_square_value': chi_square_value, 'p_value': p_value}

            # Perform Kaiser–Meyer–Olkin (KMO) test
            kmo_all, kmo_model=calculate_kmo(data)
            kmo_test = {'kmo_all': list(kmo_all), 'kmo_model': kmo_model}

            # Perform correlation analysis
            correlation_matrix = data.corr(method=corr_method)

            # Convert the correlation matrix to a dictionary
            correlation_dict = correlation_matrix.to_dict()

            # Form json output
            json_data = {
                'corr_method': corr_method,
                'percentage_of_columns_with_outliers': outliers_percentage,
                'bartletts_test': bartletts_test,
                'kmo_test': kmo_test,
                'corr_mtx': correlation_dict
            }

            # Save the results to a JSON file
            with open(output_file, 'w') as f:
                json.dump(json_data, f)
        else:
            json_data = {
                'error': 'Data has non-numeric columns',
                'columns': invalid_columns
            }
        with open(output_file, 'w') as f:
            json.dump(json_data, f)
    else:
        json_data = {
            'error': 'Filename extension is not csv'
        }
        with open(output_file, 'w') as f:
            json.dump(json_data, f)


if __name__ == '__main__':
    perform_correlation_analysis('test_data.csv', 'corr_results.json')
