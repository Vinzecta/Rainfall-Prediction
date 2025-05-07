import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader
from torch import nn


rain_type_df = pd.read_csv("../processed/rain_type.csv", parse_dates=["Date Time"], index_col="Date Time")
rain_type_df = rain_type_df.drop(columns=['Rain_Rate (mm/h)'])
rain_type_df = rain_type_df.drop(columns=['rain (mm)'])
rain_type_df = rain_type_df.drop(columns=['raining (s)'])

# mask = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
# selected_feature = [index for index, value in enumerate(mask) if value == 1]
# # Select columns where chromosome is 1
# new_X = rain_type_df.iloc[:, selected_feature].values

X = rain_type_df.drop(columns=[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
]).values

y = rain_type_df[[
    "Rain_Type_Cloudburst", "Rain_Type_Heavy_Rain", "Rain_Type_Moderate_Rain",
    "Rain_Type_No_Rain", "Rain_Type_Shower", "Rain_Type_Very_Heavy_Rain", "Rain_Type_Weak_Rain"
]].values

# new_X = new_X.astype(np.float32)
X = X.astype(np.float32)
y = y.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_ds = TensorDataset(X_train, y_train)
batch_size = 10

torch.manual_seed(324)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

hidden_units = [128, 64, 32, 16]
input_size = X.shape[1]
all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 7))
# all_layers.append(nn.Softmax(dim=1))
model = nn.Sequential(*all_layers)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

############################################ Training ############################################

torch.manual_seed(43)
num_epochs = 50
for epoch in range(num_epochs):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # is_correct = (
        #     torch.argmax(pred, dim=1) == y_batch
        # ).float()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1)).float()
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train /= len(train_dl.dataset)
    print(f'Epoch {epoch} Accuracy ' f'{accuracy_hist_train}')

pred = model(X_test)
is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1)).float()
print(f'Test accuracy: {is_correct.mean()}')

# 5 chosen feature: 
# All feature: Test accuracy: 0.8668941855430603
