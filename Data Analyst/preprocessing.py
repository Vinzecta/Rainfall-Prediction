# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# data = pd.read_csv("../Code/Dataset/mpi_roof.csv", encoding='latin-1')
# data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
# data = data.set_index("Date Time")

# is_nan = data.isnull().values.any()
# if (is_nan):
#     print("There exist null values")
# else:
#     print("No null values")

# #Check if exist a case when having no record of rain (0mm) but the raining(s)
# checker =  not data.loc[(data['rain (mm)'] == 0) & (data['raining (s)'] > 0), ['rain (mm)', 'raining (s)']].empty
# if (checker):
#     print("There exist a case when having no record of rain (0mm) but the raining(s)")
# else:
#     print("There is no case when having no record of rain (0mm) but the raining(s)")


# #Check if exist a case when having no record of raining (s) but rain
# checker_2 = not data.loc[(data['rain (mm)'] > 0) & (data['raining (s)'] == 0), ['rain (mm)', 'raining (s)']].empty
# if (checker_2):
#     print("There exist a case")
# else:
#     print("There is no case")

# resample_data = data.resample('6h').mean()

# #Preprocessing data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(resample_data)
# scaled_data = pd.DataFrame(scaled_data, columns = resample_data.columns, index = resample_data.index)
# scaled_data.head()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

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

resample_data = data.resample('6h').mean()

resample_data['Rain Rate (mm/h)'] = np.where((resample_data['raining (s)'] > 0), (resample_data['rain (mm)'] * 3600) / resample_data['raining (s)'], 0)
resample_data['Is Rain'] = np.where(resample_data['Rain Rate (mm/h)'] >= 0.5, 'Yes', 'No')

rain_condition = ['No Rain', 'Weak Rain', 'Moderate Rain', 'Heavy Rain', 'Very Heavy Rain', 'Shower', 'Cloudburst']
rain_rate = [resample_data['Rain Rate (mm/h)'] < 0.5,
             (resample_data['Rain Rate (mm/h)'] >= 0.5) & (resample_data['Rain Rate (mm/h)'] < 2),
             (resample_data['Rain Rate (mm/h)'] >= 2) & (resample_data['Rain Rate (mm/h)'] < 6),
             (resample_data['Rain Rate (mm/h)'] >= 6) & (resample_data['Rain Rate (mm/h)'] < 10),
             (resample_data['Rain Rate (mm/h)'] >= 10) & (resample_data['Rain Rate (mm/h)'] < 18),
             (resample_data['Rain Rate (mm/h)'] >= 18) & (resample_data['Rain Rate (mm/h)'] < 30),
             resample_data['Rain Rate (mm/h)'] >= 30]
resample_data['Rain Type'] = np.select(rain_rate, rain_condition, default='Unknown')
resample_data.head()
resample_data.to_csv("ten_file.csv")

#Preprocessing data
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False)
regression_data = resample_data.drop(columns=['Is Rain', 'Rain Type'])
categorical_columns = resample_data.select_dtypes(include=['object']).columns.tolist()

one_hot_encoded = encoder.fit_transform(resample_data[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=resample_data.index)

regression_data_scaler = scaler.fit_transform(regression_data)
regression_data_df = pd.DataFrame(regression_data_scaler, columns=regression_data.columns, index=regression_data.index)

preprocessed_data = pd.concat([regression_data_df, one_hot_df], axis=1)
preprocessed_data.head()
