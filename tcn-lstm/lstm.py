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

# Splitting the dataset
train_data = rain_type_df[:training_data_len].iloc[:]
test_data = rain_type_df[training_data_len:].iloc[:]

input_hours = 12
output_hours = 1

X_train, y_train = [], []
timestamps = train_data.index

for i in range(len(train_data) - input_hours - output_hours):
    # Extract input (past 3 days)
    X_train.append(train_data.iloc[i:i+input_hours, :19])

    # Extract output (next 6 hours)
    y_train.append(train_data.iloc[i+input_hours+output_hours][19:])

# Convert to torch tensor arrays
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

# Convert to torch tensor arrays
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
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
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

class LSTMConfig1(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 512, batch_first=True)
        self.lstm4 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        return out
    

class LSTMConfig2(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 2048, batch_first=True)
        self.lstm3 = nn.LSTM(2048, 2048, batch_first=True)
        self.lstm4 = nn.LSTM(2048, 1024, batch_first=True)
        self.lstm5 = nn.LSTM(1024, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)
        out = self.fc(out[:, -1, :]) 
        return out
    
class LSTMConfig3(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 1024, batch_first=True)
        self.lstm4 = nn.LSTM(1024, 512, batch_first=True)
        self.lstm5 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)
        out = self.fc(out[:, -1, :]) 
        return out
    
class LSTMConfig4(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 1024, batch_first=True)
        self.lstm3 = nn.LSTM(1024, 1024, batch_first=True)
        self.lstm4 = nn.LSTM(1024, 512, batch_first=True)
        self.fc = nn.Linear(512, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        return out
    
class LSTMConfig5(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 512, batch_first=True)
        self.lstm4 = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        return out
    
class LSTMConfig6(nn.Module):
    def __init__(self, input_size=X.shape[1], output_size=7, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :]) 
        return out

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = X.shape[1]
num_layers = 3  # Increased number of layers
hidden_size = 64  # Increased number of hidden units
output_size = 7
dropout = 0  # Added dropout for regularization

train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)

batch_size = 10
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


## Config 1 ##
lstm1 = LSTMConfig1()
print(lstm1)
lstm1.to(device)

learning_rate = 0.001
num_epochs = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm1(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(y_batch)
        # print(pred)
        # print(torch.argmax(pred, dim=1))
        # print(torch.argmax(y_batch, dim=1))
        # print((torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1)).float())
        # print((torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1)).sum())
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1)).float()
        # print(f'is_correct: {is_correct}')
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy {accuracy_hist_train}')


# pred = lstm1(X_test)
# is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1)).float()
# print(f'Test accuracy of config 1: {is_correct.mean()}')
lstm1.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm1(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 1 Accuracy: {accuracy_hist_test}' )


## Config 2 ##
lstm2 = LSTMConfig2()
print(lstm2)
lstm2.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm2.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm2(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1))
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')


lstm2.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm2(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 2 Accuracy: {accuracy_hist_test}')

## Config 3 ##
lstm3 = LSTMConfig3()
print(lstm3)
lstm3.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm3.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm3(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1))
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')


lstm3.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm3(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 3 Accuracy: {accuracy_hist_test}')


## Config 4 ##
lstm4 = LSTMConfig4()
print(lstm4)
lstm4.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm4.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm4(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1))
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')

X_test = X_test.to(device)
y_test = y_test.to(device)

lstm4.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm4(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 4 Accuracy: {accuracy_hist_test}')

## Config 5 ##
lstm5 = LSTMConfig5()
print(lstm5)
lstm5.to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm5.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm5(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1))
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')


lstm5.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm5(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 2 Accuracy: {accuracy_hist_test}')

## Config 6 ##
lstm6 = LSTMConfig6()
print(lstm6)
lstm6.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm6.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    accuracy_hist_train = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = lstm6(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_batch, dim=1))
        accuracy_hist_train += is_correct.sum()

    accuracy_hist_train = accuracy_hist_train.float() / len(train_loader.dataset)
    print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')


lstm6.eval()
accuracy_hist_test = 0

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        pred = lstm6(x_test)

        is_correct = (torch.argmax(pred, dim=1) == torch.argmax(y_test, dim=1))
        accuracy_hist_test += is_correct.sum()

    accuracy_hist_test = accuracy_hist_test.float() / len(test_loader.dataset)
    print(f'Config 6 Accuracy: {accuracy_hist_test}')


# Questions: why it is the same for every config/model ?