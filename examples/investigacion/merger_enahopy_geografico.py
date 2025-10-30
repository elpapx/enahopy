import pandas as pd
import numpy as numpy
from enahopy.merger.geographic.merger import GeographicMerger


df_ubigeo = pd.read_excel(r'E:\papx\end_to_end_ml\nb_pr\enahopy\examples\data\UBIGEO 2022_1891 distritos.xlsx')
df = pd.read_csv(r'E:\papx\end_to_end_ml\nb_pr\enahopy\examples\investigacion\dataframe_final_2024.csv')

print(list(df_ubigeo.columns))
df_ubigeo = df_ubigeo[['IDDIST', 'NOMBDEP', 'NOMBPROV', 'NOMBDIST',]]
df_ubigeo = df_ubigeo.rename(columns={'IDDIST': 'ubigeo'})
df_ubigeo = df_ubigeo.dropna()

###
# MERGER GEOGRAFICO ENAHOPY
###
merger = GeographicMerger()
result, report = merger.merge(df, df_ubigeo, columna_union='ubigeo')
print(f"Records: {report['output_rows']}")
print(result.columns.tolist())

result.to_csv('dataframe_final_completo_geografico_2024.csv', index=False)