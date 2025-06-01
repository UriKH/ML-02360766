import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def prepare_data(training_data: pd.DataFrame, new_data: pd.DataFrame):
    new_df = new_data.copy(True)

    scaler_mm = MinMaxScaler(feature_range=(-1, 1))
    scaler_z = StandardScaler()

    mm_scale_columns = ['PCR_01', 'PCR_02', 'PCR_04', 'PCR_06', 'PCR_08']
    z_scale_columns = ['PCR_03', 'PCR_05', 'PCR_07', 'PCR_09', 'PCR_10']

    scaler_mm.fit(new_df[mm_scale_columns])
    new_df[mm_scale_columns] = scaler_mm.transform(new_df[mm_scale_columns])
    scaler_z.fit(new_df[z_scale_columns])
    new_df[z_scale_columns] = scaler_z.transform(new_df[z_scale_columns])
    return new_df