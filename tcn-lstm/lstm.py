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
training_data_len = math.ceil(len(rain_type_df) * .9)

# Splitting the dataset
train_data = rain_type_df[:training_data_len].iloc[:]
test_data = rain_type_df[training_data_len:].iloc[:]

input_hours = 28
output_hours = 4

X_train, y_train = [], []
timestamps = train_data.index

for i in range(len(train_data) - input_hours - output_hours):
    # Extract input
    X_train.append(train_data.iloc[i:i+input_hours, :19])

    # Extract output
    y_train.append(train_data.iloc[i+input_hours:i+input_hours+output_hours, 19:])

# Convert to torch tensor arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

print("Input shape:", X_train.shape)  
print("Output shape:", y_train.shape) 


X_test, y_test = [], []
timestamps = test_data.index

for i in range(len(test_data) - input_hours - output_hours):
    # Extract input
    X_test.append(test_data.iloc[i:i+input_hours, :19])

    # Extract output
    y_test.append(train_data.iloc[i+input_hours:i+input_hours+output_hours, 19:])

# Convert to torch tensor arrays
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print("Input shape:", X_test.shape)  
print("Output shape:", y_test.shape) 

input_size = X.shape[1]
output_size = 7

class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class LSTMConfig1(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 512, batch_first=True)
        self.lstm4 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size * output_hours)
        # self.fc = nn.Linear(256, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out
    

class LSTMConfig2(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 2048, batch_first=True)
        self.lstm3 = nn.LSTM(2048, 2048, batch_first=True)
        self.lstm4 = nn.LSTM(2048, 1024, batch_first=True)
        self.lstm5 = nn.LSTM(1024, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size * output_hours)
        # self.fc = nn.Linear(256, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out
    
class LSTMConfig3(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 1024, batch_first=True)
        self.lstm4 = nn.LSTM(1024, 512, batch_first=True)
        self.lstm5 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size * output_hours)
        # self.fc = nn.Linear(256, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out
    
class LSTMConfig4(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 1024, batch_first=True)
        self.lstm3 = nn.LSTM(1024, 1024, batch_first=True)
        self.lstm4 = nn.LSTM(1024, 512, batch_first=True)
        self.fc = nn.Linear(512, output_size * output_hours)
        # self.fc = nn.Linear(512, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out
    
class LSTMConfig5(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 256, batch_first=True)
        self.lstm3 = nn.LSTM(256, 512, batch_first=True)
        self.lstm4 = nn.LSTM(512, 128, batch_first=True)
        self.fc = nn.Linear(128, output_size * output_hours)
        # self.fc = nn.Linear(128, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out, _ = self.lstm4(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out
    
class LSTMConfig6(nn.Module):
    def __init__(self, input_size=input_size, output_size=output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, output_size * output_hours)
        # self.fc = nn.Linear(256, output_hours)

    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :]) 
        out = out.view(-1, output_hours, output_size)
        return out

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)



train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


# LSTMConfig1(
#   (lstm): LSTM(19, 128, batch_first=True)
#   (lstm2): LSTM(128, 512, batch_first=True)
#   (lstm3): LSTM(512, 512, batch_first=True)
#   (lstm4): LSTM(512, 256, batch_first=True)
#   (fc): Linear(in_features=256, out_features=84, bias=True)
# )
# Epoch 0 Accuracy 0.07577042281627655
# Epoch 1 Accuracy 0.10369454324245453
# Epoch 2 Accuracy 0.0017411229200661182
# Epoch 3 Accuracy 0.0017711941618472338
# Epoch 4 Accuracy 0.0016900019254535437
# Epoch 5 Accuracy 0.04825526848435402
# Epoch 6 Accuracy 0.06501996517181396
# Epoch 7 Accuracy 0.08184480667114258
# Epoch 8 Accuracy 0.07834752649068832
# Epoch 9 Accuracy 0.07737021148204803
# Epoch 10 Accuracy 0.14206841588020325
# Epoch 11 Accuracy 0.26306894421577454
# Epoch 12 Accuracy 0.21021278202533722
# Epoch 13 Accuracy 0.2551632225513458
# Epoch 14 Accuracy 0.2527214288711548
# Epoch 15 Accuracy 0.28448864817619324
# Epoch 16 Accuracy 0.32848885655403137
# Epoch 17 Accuracy 0.3242909014225006
# Epoch 18 Accuracy 0.31833380460739136
# Epoch 19 Accuracy 0.334909051656723
# Epoch 20 Accuracy 0.3463180661201477
# Epoch 21 Accuracy 0.40028685331344604
# Epoch 22 Accuracy 0.4015859365463257
# Epoch 23 Accuracy 0.39315396547317505
# Epoch 24 Accuracy 0.4049418866634369
# Epoch 25 Accuracy 0.4138159155845642
# Epoch 26 Accuracy 0.43459510803222656
# Epoch 27 Accuracy 0.4335666596889496
# Epoch 28 Accuracy 0.43260741233825684
# Epoch 29 Accuracy 0.42075634002685547
# Epoch 30 Accuracy 0.43322986364364624
# Epoch 31 Accuracy 0.42296355962753296
# Epoch 32 Accuracy 0.404211163520813
# Epoch 33 Accuracy 0.42748627066612244
# Epoch 34 Accuracy 0.42752838134765625
# Epoch 35 Accuracy 0.42151111364364624
# Epoch 36 Accuracy 0.3842017650604248
# Epoch 37 Accuracy 0.40283089876174927
# Epoch 38 Accuracy 0.39114221930503845
# Epoch 39 Accuracy 0.3986840844154358
# Epoch 40 Accuracy 0.3882042467594147
# Epoch 41 Accuracy 0.3811405301094055
# Epoch 42 Accuracy 0.3826049864292145
# Epoch 43 Accuracy 0.4062590003013611
# Epoch 44 Accuracy 0.38814711570739746
# Epoch 45 Accuracy 0.390249103307724
# Epoch 46 Accuracy 0.3795227110385895
# Epoch 47 Accuracy 0.3794204592704773
# Epoch 48 Accuracy 0.39249542355537415
# Epoch 49 Accuracy 0.3800790011882782
# Epoch 50 Accuracy 0.3969970643520355
# Epoch 51 Accuracy 0.40397360920906067
# Epoch 52 Accuracy 0.39567694067955017
# Epoch 53 Accuracy 0.35535144805908203
# Epoch 54 Accuracy 0.3673167824745178
# Epoch 55 Accuracy 0.3732077479362488
# Epoch 56 Accuracy 0.37228456139564514
# Epoch 57 Accuracy 0.38245463371276855
# Epoch 58 Accuracy 0.3827042281627655
# Epoch 59 Accuracy 0.3815043866634369
# Epoch 60 Accuracy 0.37907764315605164
# Epoch 61 Accuracy 0.38117361068725586
# Epoch 62 Accuracy 0.3786446154117584
# Epoch 63 Accuracy 0.3778868317604065
# Epoch 64 Accuracy 0.3727656900882721
# Epoch 65 Accuracy 0.37362271547317505
# Epoch 66 Accuracy 0.3745579421520233
# Epoch 67 Accuracy 0.3766930103302002
# Epoch 68 Accuracy 0.3744647204875946
# Epoch 69 Accuracy 0.39792928099632263
# Epoch 70 Accuracy 0.39219769835472107
# Epoch 71 Accuracy 0.3960348069667816
# Epoch 72 Accuracy 0.39599570631980896
# Epoch 73 Accuracy 0.3961370289325714
# Epoch 74 Accuracy 0.4066770076751709
# Epoch 75 Accuracy 0.4181702136993408
# Epoch 76 Accuracy 0.4220764636993408
# Epoch 77 Accuracy 0.4271554946899414
# Epoch 78 Accuracy 0.43296825885772705
# Epoch 79 Accuracy 0.45333850383758545
# Epoch 80 Accuracy 0.46266356110572815
# Epoch 81 Accuracy 0.4701392650604248
# Epoch 82 Accuracy 0.46003836393356323
# Epoch 83 Accuracy 0.4665728211402893
# Epoch 84 Accuracy 0.44017332792282104
# Epoch 85 Accuracy 0.44139420986175537
# Epoch 86 Accuracy 0.4544481337070465
# Epoch 87 Accuracy 0.46521061658859253
# Epoch 88 Accuracy 0.47250887751579285
# Epoch 89 Accuracy 0.4877550005912781
# Epoch 90 Accuracy 0.4888766407966614
# Epoch 91 Accuracy 0.49855056405067444
# Epoch 92 Accuracy 0.5132673978805542
# Epoch 93 Accuracy 0.5080410242080688
# Epoch 94 Accuracy 0.515354335308075
# Epoch 95 Accuracy 0.531304121017456
# Epoch 96 Accuracy 0.5392068028450012
# Epoch 97 Accuracy 0.5455939769744873
# Epoch 98 Accuracy 0.56944340467453
# Epoch 99 Accuracy 0.5856097340583801
# Config 1 Accuracy: 0.5814163684844971
# LSTMConfig2(
#   (lstm): LSTM(19, 256, batch_first=True)
#   (lstm2): LSTM(256, 2048, batch_first=True)
#   (lstm3): LSTM(2048, 2048, batch_first=True)
#   (lstm4): LSTM(2048, 1024, batch_first=True)
#   (lstm5): LSTM(1024, 256, batch_first=True)
#   (fc): Linear(in_features=256, out_features=84, bias=True)
# )
# Epoch 0 Accuracy: 0.030059179291129112
# Epoch 1 Accuracy: 0.046159304678440094
# Epoch 2 Accuracy: 0.05516262352466583
# Epoch 3 Accuracy: 0.055733975023031235
# Epoch 4 Accuracy: 0.0570962019264698
# Epoch 5 Accuracy: 0.0595921128988266
# Epoch 6 Accuracy: 0.05958910658955574
# Epoch 7 Accuracy: 0.05558963492512703
# Epoch 8 Accuracy: 0.04993624612689018
# Epoch 9 Accuracy: 0.044893305748701096
# Epoch 10 Accuracy: 0.04038563370704651
# Epoch 11 Accuracy: 0.03717102110385895
# Epoch 12 Accuracy: 0.03286181390285492
# Epoch 13 Accuracy: 0.027024993672966957
# Epoch 14 Accuracy: 0.02246319130063057
# Epoch 15 Accuracy: 0.017336051911115646
# Epoch 16 Accuracy: 0.012235973961651325
# Epoch 17 Accuracy: 0.009562644176185131
# Epoch 18 Accuracy: 0.0063901315443217754
# Epoch 19 Accuracy: 0.004023527726531029
# Epoch 20 Accuracy: 0.0037198083009570837
# Epoch 21 Accuracy: 0.00372882978990674
# Epoch 22 Accuracy: 0.0037498795427381992
# Epoch 23 Accuracy: 0.0037949865218251944
# Epoch 24 Accuracy: 0.0039724064990878105
# Epoch 25 Accuracy: 0.004005484748631716
# Epoch 26 Accuracy: 0.0040114992298185825
# Epoch 27 Accuracy: 0.004014506004750729
# Epoch 28 Accuracy: 0.004014506004750729
# Epoch 29 Accuracy: 0.004014506004750729
# Epoch 30 Accuracy: 0.004014506004750729
# Epoch 31 Accuracy: 0.004014506004750729
# Epoch 32 Accuracy: 0.004014506004750729
# Epoch 33 Accuracy: 0.004014506004750729
# Epoch 34 Accuracy: 0.004014506004750729
# Epoch 35 Accuracy: 0.004014506004750729
# Epoch 36 Accuracy: 0.004014506004750729
# Epoch 37 Accuracy: 0.004014506004750729
# Epoch 38 Accuracy: 0.004014506004750729
# Epoch 39 Accuracy: 0.004014506004750729
# Epoch 40 Accuracy: 0.004014506004750729
# Epoch 41 Accuracy: 0.004014506004750729
# Epoch 42 Accuracy: 0.004014506004750729
# Epoch 43 Accuracy: 0.004014506004750729
# Epoch 44 Accuracy: 0.004014506004750729
# Epoch 45 Accuracy: 0.004014506004750729
# Epoch 46 Accuracy: 0.004014506004750729
# Epoch 47 Accuracy: 0.004014506004750729
# Epoch 48 Accuracy: 0.004014506004750729
# Epoch 49 Accuracy: 0.004014506004750729
# Epoch 50 Accuracy: 0.004014506004750729
# Epoch 51 Accuracy: 0.004014506004750729
# Epoch 52 Accuracy: 0.004014506004750729
# Epoch 53 Accuracy: 0.004014506004750729
# Epoch 54 Accuracy: 0.004014506004750729
# Epoch 55 Accuracy: 0.004014506004750729
# Epoch 56 Accuracy: 0.004014506004750729
# Epoch 57 Accuracy: 0.004014506004750729
# Epoch 58 Accuracy: 0.004014506004750729
# Epoch 59 Accuracy: 0.004014506004750729
# Epoch 60 Accuracy: 0.004014506004750729
# Epoch 61 Accuracy: 0.004014506004750729
# Epoch 62 Accuracy: 0.004014506004750729
# Epoch 63 Accuracy: 0.004014506004750729
# Epoch 64 Accuracy: 0.004014506004750729
# Epoch 65 Accuracy: 0.004014506004750729
# Epoch 66 Accuracy: 0.004014506004750729
# Epoch 67 Accuracy: 0.004014506004750729
# Epoch 68 Accuracy: 0.004014506004750729
# Epoch 69 Accuracy: 0.004014506004750729
# Epoch 70 Accuracy: 0.004014506004750729
# Epoch 71 Accuracy: 0.004014506004750729
# Epoch 72 Accuracy: 0.004014506004750729
# Epoch 73 Accuracy: 0.004014506004750729
# Epoch 74 Accuracy: 0.004014506004750729
# Epoch 75 Accuracy: 0.004014506004750729
# Epoch 76 Accuracy: 0.004014506004750729
# Epoch 77 Accuracy: 0.004014506004750729
# Epoch 78 Accuracy: 0.004014506004750729
# Epoch 79 Accuracy: 0.004014506004750729
# Epoch 80 Accuracy: 0.004014506004750729
# Epoch 81 Accuracy: 0.004014506004750729
# Epoch 82 Accuracy: 0.004014506004750729
# Epoch 83 Accuracy: 0.004014506004750729
# Epoch 84 Accuracy: 0.004014506004750729
# Epoch 85 Accuracy: 0.004014506004750729
# Epoch 86 Accuracy: 0.004014506004750729
# Epoch 87 Accuracy: 0.004014506004750729
# Epoch 88 Accuracy: 0.004014506004750729
# Epoch 89 Accuracy: 0.004014506004750729
# Epoch 90 Accuracy: 0.004014506004750729
# Epoch 91 Accuracy: 0.004014506004750729
# Epoch 92 Accuracy: 0.004014506004750729
# Epoch 93 Accuracy: 0.004014506004750729
# Epoch 94 Accuracy: 0.004014506004750729
# Epoch 95 Accuracy: 0.004014506004750729
# Epoch 96 Accuracy: 0.004014506004750729
# Epoch 97 Accuracy: 0.004014506004750729
# Epoch 98 Accuracy: 0.004014506004750729
# Epoch 99 Accuracy: 0.004014506004750729
# Config 2 Accuracy: 0.0027111403178423643
# LSTMConfig3(
#   (lstm): LSTM(19, 256, batch_first=True)
#   (lstm2): LSTM(256, 512, batch_first=True)
#   (lstm3): LSTM(512, 1024, batch_first=True)
#   (lstm4): LSTM(1024, 512, batch_first=True)
#   (lstm5): LSTM(512, 256, batch_first=True)
#   (fc): Linear(in_features=256, out_features=84, bias=True)
# )
# Epoch 0 Accuracy: 0.034554824233055115
# Epoch 1 Accuracy: 0.0132523812353611
# Epoch 2 Accuracy: 0.002489896025508642
# Epoch 3 Accuracy: 0.002640252001583576
# Epoch 4 Accuracy: 0.0027996294666081667
# Epoch 5 Accuracy: 0.0029319426976144314
# Epoch 6 Accuracy: 0.0028417292051017284
# Epoch 7 Accuracy: 0.0029349499382078648
# Epoch 8 Accuracy: 0.0032085978891700506
# Epoch 9 Accuracy: 0.007174990139901638
# Epoch 10 Accuracy: 0.00583982840180397
# Epoch 11 Accuracy: 0.07968268543481827
# Epoch 12 Accuracy: 0.3012142777442932
# Epoch 13 Accuracy: 0.2860764265060425
# Epoch 14 Accuracy: 0.2793554961681366
# Epoch 15 Accuracy: 0.4463709890842438
# Epoch 16 Accuracy: 0.42776593565940857
# Epoch 17 Accuracy: 0.37289199233055115
# Epoch 18 Accuracy: 0.3628632426261902
# Epoch 19 Accuracy: 0.40142056345939636
# Epoch 20 Accuracy: 0.4099337160587311
# Epoch 21 Accuracy: 0.4052095115184784
# Epoch 22 Accuracy: 0.4242205321788788
# Epoch 23 Accuracy: 0.4439622759819031
# Epoch 24 Accuracy: 0.45098692178726196
# Epoch 25 Accuracy: 0.46423327922821045
# Epoch 26 Accuracy: 0.45783114433288574
# Epoch 27 Accuracy: 0.4511883854866028
# Epoch 28 Accuracy: 0.4546074867248535
# Epoch 29 Accuracy: 0.44558313488960266
# Epoch 30 Accuracy: 0.44489148259162903
# Epoch 31 Accuracy: 0.44444042444229126
# Epoch 32 Accuracy: 0.4564448595046997
# Epoch 33 Accuracy: 0.4537474513053894
# Epoch 34 Accuracy: 0.45434287190437317
# Epoch 35 Accuracy: 0.4595000743865967
# Epoch 36 Accuracy: 0.447787344455719
# Epoch 37 Accuracy: 0.4529716372489929
# Epoch 38 Accuracy: 0.4566643536090851
# Epoch 39 Accuracy: 0.4438660740852356
# Epoch 40 Accuracy: 0.44235047698020935
# Epoch 41 Accuracy: 0.453864723443985
# Epoch 42 Accuracy: 0.4643325209617615
# Epoch 43 Accuracy: 0.5906165838241577
# Epoch 44 Accuracy: 0.5141063928604126
# Epoch 45 Accuracy: 0.5119562745094299
# Epoch 46 Accuracy: 0.4757084548473358
# Epoch 47 Accuracy: 0.4633913040161133
# Epoch 48 Accuracy: 0.46872591972351074
# Epoch 49 Accuracy: 0.4583573639392853
# Epoch 50 Accuracy: 0.46025487780570984
# Epoch 51 Accuracy: 0.4577769935131073
# Epoch 52 Accuracy: 0.4586069583892822
# Epoch 53 Accuracy: 0.4620170295238495
# Epoch 54 Accuracy: 0.4674629271030426
# Epoch 55 Accuracy: 0.48270002007484436
# Epoch 56 Accuracy: 0.49058768153190613
# Epoch 57 Accuracy: 0.4897487163543701
# Epoch 58 Accuracy: 0.46717727184295654
# Epoch 59 Accuracy: 0.4627447724342346
# Epoch 60 Accuracy: 0.46898454427719116
# Epoch 61 Accuracy: 0.4539790153503418
# Epoch 62 Accuracy: 0.45099595189094543
# Epoch 63 Accuracy: 0.45461952686309814
# Epoch 64 Accuracy: 0.45383766293525696
# Epoch 65 Accuracy: 0.4568357765674591
# Epoch 66 Accuracy: 0.5079989433288574
# Epoch 67 Accuracy: 0.47431617975234985
# Epoch 68 Accuracy: 0.47216907143592834
# Epoch 69 Accuracy: 0.4580175578594208
# Epoch 70 Accuracy: 0.47640612721443176
# Epoch 71 Accuracy: 0.47138121724128723
# Epoch 72 Accuracy: 0.48179787397384644
# Epoch 73 Accuracy: 0.6673341989517212
# Epoch 74 Accuracy: 0.5277045965194702
# Epoch 75 Accuracy: 0.4595211446285248
# Epoch 76 Accuracy: 0.46887028217315674
# Epoch 77 Accuracy: 0.467721551656723
# Epoch 78 Accuracy: 0.4578281342983246
# Epoch 79 Accuracy: 0.5389782786369324
# Epoch 80 Accuracy: 0.7433121204376221
# Epoch 81 Accuracy: 0.5250763893127441
# Epoch 82 Accuracy: 0.4639115333557129
# Epoch 83 Accuracy: 0.4559516906738281
# Epoch 84 Accuracy: 0.46625709533691406
# Epoch 85 Accuracy: 0.6690092086791992
# Epoch 86 Accuracy: 0.7568501830101013
# Epoch 87 Accuracy: 0.7428280115127563
# Epoch 88 Accuracy: 0.7027009725570679
# Epoch 89 Accuracy: 0.5203431844711304
# Epoch 90 Accuracy: 0.5087206363677979
# Epoch 91 Accuracy: 0.4930835962295532
# Epoch 92 Accuracy: 0.4714714288711548
# Epoch 93 Accuracy: 0.4597376585006714
# Epoch 94 Accuracy: 0.4736636281013489
# Epoch 95 Accuracy: 0.4684312343597412
# Epoch 96 Accuracy: 0.47575056552886963
# Epoch 97 Accuracy: 0.4546796679496765
# Epoch 98 Accuracy: 0.4601556360721588
# Epoch 99 Accuracy: 0.46327102184295654
# Config 3 Accuracy: 0.4673841595649719
# LSTMConfig4(
#   (lstm): LSTM(19, 256, batch_first=True)
#   (lstm2): LSTM(256, 1024, batch_first=True)
#   (lstm3): LSTM(1024, 1024, batch_first=True)
#   (lstm4): LSTM(1024, 512, batch_first=True)
#   (fc): Linear(in_features=512, out_features=84, bias=True)
# )
# Epoch 0 Accuracy: 0.025864245370030403
# Epoch 1 Accuracy: 0.10859615355730057
# Epoch 2 Accuracy: 0.10985312610864639
# Epoch 3 Accuracy: 0.20983989536762238
# Epoch 4 Accuracy: 0.22244875133037567
# Epoch 5 Accuracy: 0.24047644436359406
# Epoch 6 Accuracy: 0.22207285463809967
# Epoch 7 Accuracy: 0.2243642807006836
# Epoch 8 Accuracy: 0.2109585404396057
# Epoch 9 Accuracy: 0.22525440156459808
# Epoch 10 Accuracy: 0.24425338208675385
# Epoch 11 Accuracy: 0.2548023760318756
# Epoch 12 Accuracy: 0.26000767946243286
# Epoch 13 Accuracy: 0.26067227125167847
# Epoch 14 Accuracy: 0.2595566213130951
# Epoch 15 Accuracy: 0.14362910389900208
# Epoch 16 Accuracy: 0.15951873362064362
# Epoch 17 Accuracy: 0.1918693333864212
# Epoch 18 Accuracy: 0.2159503698348999
# Epoch 19 Accuracy: 0.2124560922384262
# Epoch 20 Accuracy: 0.20512773096561432
# Epoch 21 Accuracy: 0.1808301955461502
# Epoch 22 Accuracy: 0.18864871561527252
# Epoch 23 Accuracy: 0.18406887352466583
# Epoch 24 Accuracy: 0.15685442090034485
# Epoch 25 Accuracy: 0.13534750044345856
# Epoch 26 Accuracy: 0.13728709518909454
# Epoch 27 Accuracy: 0.17445510625839233
# Epoch 28 Accuracy: 0.16709667444229126
# Epoch 29 Accuracy: 0.18780070543289185
# Epoch 30 Accuracy: 0.19029060006141663
# Epoch 31 Accuracy: 0.22363054752349854
# Epoch 32 Accuracy: 0.2709566056728363
# Epoch 33 Accuracy: 0.292692095041275
# Epoch 34 Accuracy: 0.29565709829330444
# Epoch 35 Accuracy: 0.29171475768089294
# Epoch 36 Accuracy: 0.2981860935688019
# Epoch 37 Accuracy: 0.3078329265117645
# Epoch 38 Accuracy: 0.32353612780570984
# Epoch 39 Accuracy: 0.3012383282184601
# Epoch 40 Accuracy: 0.2741381525993347
# Epoch 41 Accuracy: 0.30247727036476135
# Epoch 42 Accuracy: 0.2913208305835724
# Epoch 43 Accuracy: 0.30511149764060974
# Epoch 44 Accuracy: 0.3083471655845642
# Epoch 45 Accuracy: 0.295212060213089
# Epoch 46 Accuracy: 0.2890264093875885
# Epoch 47 Accuracy: 0.28865954279899597
# Epoch 48 Accuracy: 0.2762070596218109
# Epoch 49 Accuracy: 0.2578936815261841
# Epoch 50 Accuracy: 0.32186415791511536
# Epoch 51 Accuracy: 0.2621728181838989
# Epoch 52 Accuracy: 0.2473747730255127
# Epoch 53 Accuracy: 0.2521620988845825
# Epoch 54 Accuracy: 0.23886463046073914
# Epoch 55 Accuracy: 0.3238218128681183
# Epoch 56 Accuracy: 0.29813799262046814
# Epoch 57 Accuracy: 0.26626551151275635
# Epoch 58 Accuracy: 0.26278024911880493
# Epoch 59 Accuracy: 0.236922025680542
# Epoch 60 Accuracy: 0.24650271236896515
# Epoch 61 Accuracy: 0.24123723804950714
# Epoch 62 Accuracy: 0.24622304737567902
# Epoch 63 Accuracy: 0.2353222370147705
# Epoch 64 Accuracy: 0.25176817178726196
# Epoch 65 Accuracy: 0.25938522815704346
# Epoch 66 Accuracy: 0.2740750014781952
# Epoch 67 Accuracy: 0.2747606337070465
# Epoch 68 Accuracy: 0.28026968240737915
# Epoch 69 Accuracy: 0.27967727184295654
# Epoch 70 Accuracy: 0.2809913754463196
# Epoch 71 Accuracy: 0.28162887692451477
# Epoch 72 Accuracy: 0.28871065378189087
# Epoch 73 Accuracy: 0.29214179515838623
# Epoch 74 Accuracy: 0.2996234893798828
# Epoch 75 Accuracy: 0.2974703907966614
# Epoch 76 Accuracy: 0.2970854938030243
# Epoch 77 Accuracy: 0.2964900732040405
# Epoch 78 Accuracy: 0.3084554076194763
# Epoch 79 Accuracy: 0.3168392777442932
# Epoch 80 Accuracy: 0.33427757024765015
# Epoch 81 Accuracy: 0.34121498465538025
# Epoch 82 Accuracy: 0.33771470189094543
# Epoch 83 Accuracy: 0.3618648946285248
# Epoch 84 Accuracy: 0.3556852638721466
# Epoch 85 Accuracy: 0.36190998554229736
# Epoch 86 Accuracy: 0.3667033314704895
# Epoch 87 Accuracy: 0.38289669156074524
# Epoch 88 Accuracy: 0.39420947432518005
# Epoch 89 Accuracy: 0.392723947763443
# Epoch 90 Accuracy: 0.3925856351852417
# Epoch 91 Accuracy: 0.4127483665943146
# Epoch 92 Accuracy: 0.41709667444229126
# Epoch 93 Accuracy: 0.4147571325302124
# Epoch 94 Accuracy: 0.4244520962238312
# Epoch 95 Accuracy: 0.43477553129196167
# Epoch 96 Accuracy: 0.44177311658859253
# Epoch 97 Accuracy: 0.44124385714530945
# Epoch 98 Accuracy: 0.436820387840271
# Epoch 99 Accuracy: 0.4391719400882721
# Config 4 Accuracy: 0.37054991722106934
# LSTMConfig5(
#   (lstm): LSTM(19, 64, batch_first=True)
#   (lstm2): LSTM(64, 256, batch_first=True)
#   (lstm3): LSTM(256, 512, batch_first=True)
#   (lstm4): LSTM(512, 128, batch_first=True)
#   (fc): Linear(in_features=128, out_features=84, bias=True)
# )
# Epoch 0 Accuracy: 0.058969639241695404
# Epoch 1 Accuracy: 0.16549088060855865
# Epoch 2 Accuracy: 0.19525234401226044
# Epoch 3 Accuracy: 0.20765672624111176
# Epoch 4 Accuracy: 0.21328003704547882
# Epoch 5 Accuracy: 0.21126225590705872
# Epoch 6 Accuracy: 0.2137521654367447
# Epoch 7 Accuracy: 0.06668290495872498
# Epoch 8 Accuracy: 0.07454953342676163
# Epoch 9 Accuracy: 0.09197279065847397
# Epoch 10 Accuracy: 0.09123604744672775
# Epoch 11 Accuracy: 0.10214588046073914
# Epoch 12 Accuracy: 0.12300927937030792
# Epoch 13 Accuracy: 0.12953172624111176
# Epoch 14 Accuracy: 0.08864691108465195
# Epoch 15 Accuracy: 0.13672176003456116
# Epoch 16 Accuracy: 0.1570739448070526
# Epoch 17 Accuracy: 0.11032223701477051
# Epoch 18 Accuracy: 0.14623026549816132
# Epoch 19 Accuracy: 0.17665933072566986
# Epoch 20 Accuracy: 0.18534989655017853
# Epoch 21 Accuracy: 0.23336459696292877
# Epoch 22 Accuracy: 0.22571448981761932
# Epoch 23 Accuracy: 0.23265491425991058
# Epoch 24 Accuracy: 0.25649237632751465
# Epoch 25 Accuracy: 0.2955608665943146
# Epoch 26 Accuracy: 0.30960413813591003
# Epoch 27 Accuracy: 0.3132247030735016
# Epoch 28 Accuracy: 0.3042334020137787
# Epoch 29 Accuracy: 0.29904311895370483
# Epoch 30 Accuracy: 0.3075893521308899
# Epoch 31 Accuracy: 0.3276709020137787
# Epoch 32 Accuracy: 0.335185706615448
# Epoch 33 Accuracy: 0.33771470189094543
# Epoch 34 Accuracy: 0.3285880982875824
# Epoch 35 Accuracy: 0.3053250014781952
# Epoch 36 Accuracy: 0.2925868332386017
# Epoch 37 Accuracy: 0.2924124300479889
# Epoch 38 Accuracy: 0.2876220941543579
# Epoch 39 Accuracy: 0.2865906357765198
# Epoch 40 Accuracy: 0.2875438928604126
# Epoch 41 Accuracy: 0.27121222019195557
# Epoch 42 Accuracy: 0.2714347541332245
# Epoch 43 Accuracy: 0.27166327834129333
# Epoch 44 Accuracy: 0.2700033485889435
# Epoch 45 Accuracy: 0.2747546136379242
# Epoch 46 Accuracy: 0.2752538025379181
# Epoch 47 Accuracy: 0.2794908285140991
# Epoch 48 Accuracy: 0.2866598069667816
# Epoch 49 Accuracy: 0.28237465023994446
# Epoch 50 Accuracy: 0.29361528158187866
# Epoch 51 Accuracy: 0.29441216588020325
# Epoch 52 Accuracy: 0.2588890492916107
# Epoch 53 Accuracy: 0.35192936658859253
# Epoch 54 Accuracy: 0.3298330307006836
# Epoch 55 Accuracy: 0.32402026653289795
# Epoch 56 Accuracy: 0.33406704664230347
# Epoch 57 Accuracy: 0.3435034155845642
# Epoch 58 Accuracy: 0.362556517124176
# Epoch 59 Accuracy: 0.3781604766845703
# Epoch 60 Accuracy: 0.37563449144363403
# Epoch 61 Accuracy: 0.3747624158859253
# Epoch 62 Accuracy: 0.38406646251678467
# Epoch 63 Accuracy: 0.39745116233825684
# Epoch 64 Accuracy: 0.40363380312919617
# Epoch 65 Accuracy: 0.41289272904396057
# Epoch 66 Accuracy: 0.42940783500671387
# Epoch 67 Accuracy: 0.4303039610385895
# Epoch 68 Accuracy: 0.4411746859550476
# Epoch 69 Accuracy: 0.44757384061813354
# Epoch 70 Accuracy: 0.4576988220214844
# Epoch 71 Accuracy: 0.4708790183067322
# Epoch 72 Accuracy: 0.4789260923862457
# Epoch 73 Accuracy: 0.49863776564598083
# Epoch 74 Accuracy: 0.4985836446285248
# Epoch 75 Accuracy: 0.5110421180725098
# Epoch 76 Accuracy: 0.518508791923523
# Epoch 77 Accuracy: 0.5161572694778442
# Epoch 78 Accuracy: 0.5231818556785583
# Epoch 79 Accuracy: 0.5375108122825623
# Epoch 80 Accuracy: 0.5453653931617737
# Epoch 81 Accuracy: 0.550300121307373
# Epoch 82 Accuracy: 0.5681112408638
# Epoch 83 Accuracy: 0.5790210962295532
# Epoch 84 Accuracy: 0.5508955121040344
# Epoch 85 Accuracy: 0.5221684575080872
# Epoch 86 Accuracy: 0.5498340129852295
# Epoch 87 Accuracy: 0.5655040740966797
# Epoch 88 Accuracy: 0.5730730295181274
# Epoch 89 Accuracy: 0.5847075581550598
# Epoch 90 Accuracy: 0.560897171497345
# Epoch 91 Accuracy: 0.5749133825302124
# Epoch 92 Accuracy: 0.5949558615684509
# Epoch 93 Accuracy: 0.591539740562439
# Epoch 94 Accuracy: 0.6141412854194641
# Epoch 95 Accuracy: 0.616480827331543
# Epoch 96 Accuracy: 0.641133189201355
# Epoch 97 Accuracy: 0.6052552461624146
# Epoch 98 Accuracy: 0.6373201608657837
# Epoch 99 Accuracy: 0.666438102722168
# Config 5 Accuracy: 0.6157301068305969
# LSTMConfig6(
#   (lstm): LSTM(19, 128, batch_first=True)
#   (lstm2): LSTM(128, 512, batch_first=True)
#   (lstm3): LSTM(512, 256, batch_first=True)
#   (fc): Linear(in_features=256, out_features=84, bias=True)
# )
# Epoch 0 Accuracy: 0.15100857615470886
# Epoch 1 Accuracy: 0.24403385818004608
# Epoch 2 Accuracy: 0.2449149489402771
# Epoch 3 Accuracy: 0.2459494024515152
# Epoch 4 Accuracy: 0.24428044259548187
# Epoch 5 Accuracy: 0.23756855726242065
# Epoch 6 Accuracy: 0.23455843329429626
# Epoch 7 Accuracy: 0.2337825894355774
# Epoch 8 Accuracy: 0.23273611068725586
# Epoch 9 Accuracy: 0.23713251948356628
# Epoch 10 Accuracy: 0.22230741381645203
# Epoch 11 Accuracy: 0.213180810213089
# Epoch 12 Accuracy: 0.20615316927433014
# Epoch 13 Accuracy: 0.19582070410251617
# Epoch 14 Accuracy: 0.19394725561141968
# Epoch 15 Accuracy: 0.19421188533306122
# Epoch 16 Accuracy: 0.1894967257976532
# Epoch 17 Accuracy: 0.18731355667114258
# Epoch 18 Accuracy: 0.17650596797466278
# Epoch 19 Accuracy: 0.16947230696678162
# Epoch 20 Accuracy: 0.16467294096946716
# Epoch 21 Accuracy: 0.16154853999614716
# Epoch 22 Accuracy: 0.16149741411209106
# Epoch 23 Accuracy: 0.16219507157802582
# Epoch 24 Accuracy: 0.15770544111728668
# Epoch 25 Accuracy: 0.15901052951812744
# Epoch 26 Accuracy: 0.15648755431175232
# Epoch 27 Accuracy: 0.1529301404953003
# Epoch 28 Accuracy: 0.15546514093875885
# Epoch 29 Accuracy: 0.1542021483182907
# Epoch 30 Accuracy: 0.15594327449798584
# Epoch 31 Accuracy: 0.15549220144748688
# Epoch 32 Accuracy: 0.15772348642349243
# Epoch 33 Accuracy: 0.15971720218658447
# Epoch 34 Accuracy: 0.15897144377231598
# Epoch 35 Accuracy: 0.15721528232097626
# Epoch 36 Accuracy: 0.15791593492031097
# Epoch 37 Accuracy: 0.15945859253406525
# Epoch 38 Accuracy: 0.16330470144748688
# Epoch 39 Accuracy: 0.16092005372047424
# Epoch 40 Accuracy: 0.16342197358608246
# Epoch 41 Accuracy: 0.16242963075637817
# Epoch 42 Accuracy: 0.1628987342119217
# Epoch 43 Accuracy: 0.1645827293395996
# Epoch 44 Accuracy: 0.16119670867919922
# Epoch 45 Accuracy: 0.1620447188615799
# Epoch 46 Accuracy: 0.16383996605873108
# Epoch 47 Accuracy: 0.16870549321174622
# Epoch 48 Accuracy: 0.16579760611057281
# Epoch 49 Accuracy: 0.16619153320789337
# Epoch 50 Accuracy: 0.16740339994430542
# Epoch 51 Accuracy: 0.16592390835285187
# Epoch 52 Accuracy: 0.16366255283355713
# Epoch 53 Accuracy: 0.16765901446342468
# Epoch 54 Accuracy: 0.16485938429832458
# Epoch 55 Accuracy: 0.1651630997657776
# Epoch 56 Accuracy: 0.16748158633708954
# Epoch 57 Accuracy: 0.16856415569782257
# Epoch 58 Accuracy: 0.16831456124782562
# Epoch 59 Accuracy: 0.172280952334404
# Epoch 60 Accuracy: 0.17145399749279022
# Epoch 61 Accuracy: 0.17373940348625183
# Epoch 62 Accuracy: 0.17146001756191254
# Epoch 63 Accuracy: 0.17301470041275024
# Epoch 64 Accuracy: 0.17518283426761627
# Epoch 65 Accuracy: 0.17511065304279327
# Epoch 66 Accuracy: 0.17756447196006775
# Epoch 67 Accuracy: 0.1834343671798706
# Epoch 68 Accuracy: 0.18380123376846313
# Epoch 69 Accuracy: 0.18274874985218048
# Epoch 70 Accuracy: 0.1863211989402771
# Epoch 71 Accuracy: 0.1837681531906128
# Epoch 72 Accuracy: 0.18622498214244843
# Epoch 73 Accuracy: 0.186820387840271
# Epoch 74 Accuracy: 0.1899598240852356
# Epoch 75 Accuracy: 0.19133105874061584
# Epoch 76 Accuracy: 0.1936706006526947
# Epoch 77 Accuracy: 0.19051913917064667
# Epoch 78 Accuracy: 0.19560718536376953
# Epoch 79 Accuracy: 0.19435623288154602
# Epoch 80 Accuracy: 0.19799785315990448
# Epoch 81 Accuracy: 0.19880977272987366
# Epoch 82 Accuracy: 0.20418350398540497
# Epoch 83 Accuracy: 0.20535026490688324
# Epoch 84 Accuracy: 0.20063509047031403
# Epoch 85 Accuracy: 0.20042459666728973
# Epoch 86 Accuracy: 0.2043248414993286
# Epoch 87 Accuracy: 0.20883551239967346
# Epoch 88 Accuracy: 0.21230272948741913
# Epoch 89 Accuracy: 0.20316408574581146
# Epoch 90 Accuracy: 0.20875732600688934
# Epoch 91 Accuracy: 0.21222755312919617
# Epoch 92 Accuracy: 0.21320787072181702
# Epoch 93 Accuracy: 0.21914994716644287
# Epoch 94 Accuracy: 0.21603155136108398
# Epoch 95 Accuracy: 0.214669331908226
# Epoch 96 Accuracy: 0.2154451608657837
# Epoch 97 Accuracy: 0.22022047638893127
# Epoch 98 Accuracy: 0.22246679663658142
# Epoch 99 Accuracy: 0.22016334533691406
# Config 6 Accuracy: 0.10781575739383698

###################### More test
models = [LSTMConfig1(), LSTMConfig5()]

learning_rate = 0.001
num_epochs = 500

for model in models:
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        accuracy_hist_train = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            is_correct = (torch.argmax(pred, dim=2) == torch.argmax(y_batch, dim=2))
            accuracy_hist_train += is_correct.sum()

        accuracy_hist_train = accuracy_hist_train.float() / (len(train_loader.dataset) * output_hours)
        print(f'Epoch {epoch} Accuracy: {accuracy_hist_train}')


    model.eval()
    accuracy_hist_test = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            pred = model(x_test)

            is_correct = (torch.argmax(pred, dim=2) == torch.argmax(y_test, dim=2))
            accuracy_hist_test += is_correct.sum()

        accuracy_hist_test = accuracy_hist_test.float() / (len(test_loader.dataset) * output_hours)
        print(f'{model.__class__.__name__} Accuracy: {accuracy_hist_test}')
    
    torch.save(model, f'{model.__class__.__name__}.pt')
    