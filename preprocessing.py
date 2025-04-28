import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./Code/Dataset/mpi_roof.csv", encoding='latin-1')
data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
data = data.set_index("Date Time")
# data.head()

data_30_min = data.resample('30T').mean()
data_1_hr = data.resample('1H').mean()
data_12_hr = data.resample('12H').mean()
data_1_day = data.resample('D').mean()

scaler = StandardScaler()

# 30 min data
scaled_30_min = scaler.fit_transform(data_30_min)
scaled_30_min_df = pd.DataFrame(scaled_30_min, columns = data_30_min.columns, index = data_30_min.index)

# 1hr data
scaled_1_hr = scaler.fit_transform(data_1_hr)
scaled_1_hr_df = pd.DataFrame(scaled_1_hr, columns = data_1_hr.columns, index = data_1_hr.index)

# 12 hr data
scaled_12_hr = scaler.fit_transform(data_12_hr)
scaled_12_hr_df = pd.DataFrame(scaled_12_hr, columns = data_12_hr.columns, index = data_12_hr.index)

# 1 day data
scaled_day = scaler.fit_transform(data_1_day)
scaled_day_df = pd.DataFrame(scaled_day, columns = data_1_day.columns, index = data_1_day.index)

# Save processed data
scaled_30_min_df.to_csv("./processed/scaled_30_min.csv")
scaled_1_hr_df.to_csv("./processed/scaled_1_hr.csv")
scaled_12_hr_df.to_csv("./processed/scaled_12_hr.csv")
scaled_day_df.to_csv("./processed/scaled_day.csv")