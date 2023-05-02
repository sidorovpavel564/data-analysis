import mimetypes
import pandas as pd
from scipy import stats
import json


def filename_extension_is_csv(file_path):
    mimestart = mimetypes.guess_type(file_path)[0]
    if mimestart != None:
        mimestart = mimestart.split("/")[1]
        if mimestart == "csv":
            return True
    return False


def perform_correlation_analysis(csv_file_path, output_file):
    if filename_extension_is_csv(csv_file_path):
        # Read the CSV file into a Pandas DataFrame
        data = pd.read_csv(csv_file_path)

        # Remove missing values
        data.dropna(inplace=True)

        # Check variables types
        data_is_numeric = all(data.dtypes.apply(lambda x: x in [int, float]))

        if data_is_numeric:

            # Check data normal distribution and define corr method
            alpha = 1e-3
            threshold = 0.7
            p_values = stats.normaltest(data)[1]
            count = sum(value < alpha for value in p_values)
            corr_method = 'pearson'
            if count / len(p_values) > threshold:
                corr_method = 'kendall'

            # Perform correlation analysis
            correlation_matrix = data.corr(method=corr_method)

            # Convert the correlation matrix to a dictionary
            correlation_dict = correlation_matrix.to_dict()

            json_data = {
                'corr_method': corr_method,
                'corr_mtx': correlation_dict,
            }

            # Save the results to a JSON file
            with open(output_file, 'w') as f:
                json.dump(json_data, f)
        else:
            json_data = {
                'error': 'Data has non-numeric columns'
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
    perform_correlation_analysis('test_data.csv', 'results.json')
