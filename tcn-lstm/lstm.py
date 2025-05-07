import torch
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torch import nn


rain_type_df = pd.read_csv("../processed/rain_type.csv", parse_dates=["Date Time"], index_col="Date Time")
rain_type_df = rain_type_df.drop(columns=['Rain_Rate (mm/h)'])
rain_type_df = rain_type_df.drop(columns=['rain (mm)'])
rain_type_df = rain_type_df.drop(columns=['raining (s)'])

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

# Train test split
training_data_len = math.ceil(len(rain_type_df) * .8)
print(training_data_len)

# Splitting the dataset
train_data = rain_type_df[:training_data_len].iloc[:]
test_data = rain_type_df[training_data_len:].iloc[:]
print(train_data.shape, test_data.shape)

input_hours = 12
output_hours = 1

X_train, y_train = [], []
timestamps = train_data.index

for i in range(len(train_data) - input_hours - output_hours):
    # Extract input (past 3 days)
    X_train.append(train_data.iloc[i:i+input_hours, :19])

    # Extract output (next 6 hours)
    y_train.append(train_data.iloc[i+input_hours+output_hours][19:])

# Convert to NumPy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

print("Input shape:", X_train.shape)  # Expected: (samples, 12, features)
print("Output shape:", y_train.shape)  # Expected: (samples, targets)


X_test, y_test = [], []
timestamps = test_data.index

for i in range(len(test_data) - input_hours - output_hours):
    # Extract input (past 3 days)
    X_test.append(test_data.iloc[i:i+input_hours, :19])

    # Extract output (next 6 hours)
    y_test.append(test_data.iloc[i+input_hours+output_hours][19:])

# Convert to NumPy arrays
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Input shape:", X_test.shape)  # Expected: (samples, 12, features)
print("Output shape:", y_test.shape)  # Expected: (samples, targets)

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 7)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = X.shape[1]
num_layers = 3  # Increased number of layers
hidden_size = 64  # Increased number of hidden units
output_size = 7
dropout = 0.2  # Added dropout for regularization

train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


lstm = LSTM(input_size, hidden_size, num_layers, dropout)
print(lstm)
lstm.to(device)

learning_rate = 0.001
num_epochs = 50

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1)).float()
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train /= len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy ' f'{accuracy_hist_train}')

X_test = X_test.to(device)
y_test = y_test.to(device)

pred = lstm(X_test)
is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1)).float()
print(f'Test accuracy: {is_correct.mean()}')
# seq_length = 8
# chunk_size = seq_length + 1
# weather_chunk = [rain_type_df[i:i+chunk_size] for i in range(len(rain_type_df) - chunk_size)]

# class WeatherDataset(Dataset):
#     def __init__(self, chunks):
#         self.chunks = chunks
    
#     def __len__(self):
#         return len(self.chunks)
    
#     def __getitem__(self, index):
#         chunks = self.chunks[index]
#         return chunks[:-1], chunks[-1]
    
# weather_dataset = WeatherDataset(torch.tensor(weather_chunk))
# for i, (seq, target) in enumerate(weather_dataset):
#     print(' Input (X): ', repr(''.join(rain_type_df[seq][0])))
#     print(' Target (Y): ', repr(''.join(rain_type_df[target][0])))
#     print()
#     if i == 1:
#         break