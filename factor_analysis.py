import json

import pandas as pd
from factor_analyzer import FactorAnalyzer


def form_loadings_df(factor_loadings, corr_mtx: pd.DataFrame, n_factors: int) -> pd.DataFrame:
    # to easier dump to json
    columns = [f'Factor {n}' for n in range(1, n_factors+1)]
    index = corr_mtx.index.values.tolist()
    df_loadings = pd.DataFrame(factor_loadings, columns=columns, index=index)
    return df_loadings


def form_variance_df(factor_variance, n_factors: int) -> pd.DataFrame:
    # to easier dump to json
    columns = [f'Factor {n}' for n in range(1, n_factors+1)]
    index = ['SS Loadings', 'Proportional var', 'Cumulative var']
    df_factor_variance = pd.DataFrame(
        factor_variance, columns=columns, index=index)
    return df_factor_variance


def get_result(loadings_df: pd.DataFrame) -> dict:
    # forms a dict of variables included in each factor
    result = {}
    for factor, loading in loadings_df.items():
        result[factor] = [key for key, value in loading.items() if (
            value > 0.3 or value < -0.3)]
    return result


def perform_factor_analysis(corr_mtx: pd.DataFrame) -> None:
    # Performs factor analysis. Saves result to json.
    n_factors = 4
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', is_corr_matrix=True)
    fa.fit(corr_mtx)
    factor_loadings = fa.loadings_
    factor_variance = fa.get_factor_variance()

    loadings_df = form_loadings_df(factor_loadings, corr_mtx, n_factors)
    variance_df = form_variance_df(factor_variance, n_factors)
    result = get_result(loadings_df)

    json_data = {
        'result': result,
        'factor_loadings': loadings_df.to_dict(),
        'factor_variance': variance_df.to_dict()
    }

    with open('fa_results.json', 'w') as json_file:
        json.dump(json_data, json_file)


if __name__ == '__main__':
    with open('corr_results.json') as json_file:
        data_json = json.load(json_file)

    corr_mtx = pd.DataFrame(data_json['corr_mtx'])
    perform_factor_analysis(corr_mtx)
