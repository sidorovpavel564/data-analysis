import json

import pandas as pd
from factor_analyzer import FactorAnalyzer


def perform_factor_analysis(corr_mtx):
    fa = FactorAnalyzer(n_factors=5, rotation='varimax')
    fa.fit(corr_mtx)
    factor_loadings = fa.loadings_
    factor_variance = fa.get_factor_variance()

    # print(factor_loadings)
    # print(factor_variance)
    print(fa.corr_)

    json_data = {
        # 'factor_loadings': factor_loadings.tolist(),
        # 'factor_variance': factor_variance.tolist()
    }

    with open('fa_results.json', 'w') as json_file:
        json.dump(json_data, json_file)


if __name__ == '__main__':
    with open('corr_results.json') as json_file:
        data_json = json.load(json_file)

    data = pd.read_csv('test_data.csv')
    perform_factor_analysis(data)
