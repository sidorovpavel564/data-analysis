import pandas as pd

data = pd.read_csv('PerfIndicators_RusUniversities_01012018/data.csv', delimiter=';')
data.drop(['federal_district', 'federal_district_short', 'region_code',
       'region_name', 'okato', 'id', 'name', 'name_short', 'year'], axis=1, inplace=True)
data.dropna(inplace=True)

data.to_csv('test_data.csv', index=False)
