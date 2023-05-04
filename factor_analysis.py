import json

import pandas as pd
from factor_analyzer import FactorAnalyzer


def perform_factor_analysis(corr_mtx):
    fa = FactorAnalyzer(n_factors=5, rotation='varimax', is_corr_matrix=True)
    fa.fit(corr_mtx)
    factor_loadings = fa.loadings_
    factor_variance = fa.get_factor_variance()

    json_data = {
        'factor_loadings': factor_loadings,
        'factor_variance': factor_variance
    }

    with open('fa_results.json', 'w') as json_file:
        json.dump(json_data, json_file)


if __name__ == '__main__':
    with open('corr_results.json') as json_file:
        data_json = json.load(json_file)

    corr_mtx = pd.DataFrame(data_json['corr_mtx'])
    perform_factor_analysis(corr_mtx)
