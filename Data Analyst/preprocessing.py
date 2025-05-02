import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../Code/Dataset/mpi_roof.csv", encoding='latin-1')
data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
data = data.set_index("Date Time")

is_nan = data.isnull().values.any()
if (is_nan):
    print("There exist null values")
else:
    print("No null values")

#Check if exist a case when having no record of rain (0mm) but the raining(s)
checker =  not data.loc[(data['rain (mm)'] == 0) & (data['raining (s)'] > 0), ['rain (mm)', 'raining (s)']].empty
if (checker):
    print("There exist a case when having no record of rain (0mm) but the raining(s)")
else:
    print("There is no case when having no record of rain (0mm) but the raining(s)")


#Check if exist a case when having no record of raining (s) but rain
checker_2 = not data.loc[(data['rain (mm)'] > 0) & (data['raining (s)'] == 0), ['rain (mm)', 'raining (s)']].empty
if (checker_2):
    print("There exist a case")
else:
    print("There is no case")

data['Is Rain'] = np.where((data['rain (mm)'] > 0) & (data['raining (s)'] > 0), 1, 0)
data.head()


#Resample data
def round_mean(value):
    if 0 < value < 1:
        return 1
    return value

resample_data = data.resample('6h').mean()
resample_data['Is Rain'] = resample_data['Is Rain'].apply(round_mean)


#Preprocessing data
scaler = StandardScaler()
new_data = resample_data.drop(columns=['Is Rain'])
scaled_data = scaler.fit_transform(new_data)
scaled_data = pd.DataFrame(scaled_data, columns=new_data.columns, index=new_data.index)
scaled_data['Is Rain'] = resample_data['Is Rain']
scaled_data.head()