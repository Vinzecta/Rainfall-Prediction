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
output_hours = 12

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

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break
###################### More test
models = [LSTMConfig1(), LSTMConfig2(), LSTMConfig3(), LSTMConfig4(), LSTMConfig5(), LSTMConfig6()]

learning_rate = 0.001
num_epochs = 150

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
    

# Epoch 0 Accuracy: 0.09605345129966736
# Epoch 1 Accuracy: 0.19310827553272247
# Epoch 2 Accuracy: 0.22412070631980896
# Epoch 3 Accuracy: 0.16089299321174622
# Epoch 4 Accuracy: 0.12185154110193253
# Epoch 5 Accuracy: 0.24988873302936554
# Epoch 6 Accuracy: 0.2978823781013489
# Epoch 7 Accuracy: 0.3290962874889374
# Epoch 8 Accuracy: 0.33290931582450867
# Epoch 9 Accuracy: 0.3432959020137787
# Epoch 10 Accuracy: 0.3510061800479889
# Epoch 11 Accuracy: 0.3588908314704895
# Epoch 12 Accuracy: 0.3701675534248352
# Epoch 13 Accuracy: 0.37888818979263306
# Epoch 14 Accuracy: 0.39559876918792725
# Epoch 15 Accuracy: 0.4065326452255249
# Epoch 16 Accuracy: 0.42379653453826904
# Epoch 17 Accuracy: 0.4445727467536926
# Epoch 18 Accuracy: 0.4598308503627777
# Epoch 19 Accuracy: 0.4716518521308899
# Epoch 20 Accuracy: 0.49717026948928833
# Epoch 21 Accuracy: 0.5145845413208008
# Epoch 22 Accuracy: 0.5303057432174683
# Epoch 23 Accuracy: 0.544881284236908
# Epoch 24 Accuracy: 0.5726310014724731
# Epoch 25 Accuracy: 0.5867915153503418
# Epoch 26 Accuracy: 0.5941318869590759
# Epoch 27 Accuracy: 0.6177197694778442
# Epoch 28 Accuracy: 0.6309841275215149
# Epoch 29 Accuracy: 0.6404625773429871
# Epoch 30 Accuracy: 0.6597923636436462
# Epoch 31 Accuracy: 0.676923930644989
# Epoch 32 Accuracy: 0.6870399117469788
# Epoch 33 Accuracy: 0.7023882269859314
# Epoch 34 Accuracy: 0.7139716744422913
# Epoch 35 Accuracy: 0.7197694182395935
# Epoch 36 Accuracy: 0.7281111478805542
# Epoch 37 Accuracy: 0.737330973148346
# Epoch 38 Accuracy: 0.7468154430389404
# Epoch 39 Accuracy: 0.750971257686615
# Epoch 40 Accuracy: 0.7588499188423157
# Epoch 41 Accuracy: 0.7680457234382629
# Epoch 42 Accuracy: 0.7766490578651428
# Epoch 43 Accuracy: 0.7844225168228149
# Epoch 44 Accuracy: 0.792625904083252
# Epoch 45 Accuracy: 0.7999783158302307
# Epoch 46 Accuracy: 0.8098717331886292
# Epoch 47 Accuracy: 0.8163851499557495
# Epoch 48 Accuracy: 0.8255118131637573
# Epoch 49 Accuracy: 0.8294360637664795
# Epoch 50 Accuracy: 0.8388062715530396
# Epoch 51 Accuracy: 0.8381476998329163
# Epoch 52 Accuracy: 0.842444896697998
# Epoch 53 Accuracy: 0.847974956035614
# Epoch 54 Accuracy: 0.8520135283470154
# Epoch 55 Accuracy: 0.8518571853637695
# Epoch 56 Accuracy: 0.8574774861335754
# Epoch 57 Accuracy: 0.858069896697998
# Epoch 58 Accuracy: 0.8639909029006958
# Epoch 59 Accuracy: 0.8660086989402771
# Epoch 60 Accuracy: 0.8708080649375916
# Epoch 61 Accuracy: 0.8725612163543701
# Epoch 62 Accuracy: 0.8746691942214966
# Epoch 63 Accuracy: 0.8764764666557312
# Epoch 64 Accuracy: 0.881916344165802
# Epoch 65 Accuracy: 0.8876539468765259
# Epoch 66 Accuracy: 0.8912354111671448
# Epoch 67 Accuracy: 0.8917706608772278
# Epoch 68 Accuracy: 0.8903603553771973
# Epoch 69 Accuracy: 0.89288330078125
# Epoch 70 Accuracy: 0.8900355696678162
# Epoch 71 Accuracy: 0.8919270634651184
# Epoch 72 Accuracy: 0.8965700268745422
# Epoch 73 Accuracy: 0.9000222086906433
# Epoch 74 Accuracy: 0.9005184173583984
# Epoch 75 Accuracy: 0.9073084592819214
# Epoch 76 Accuracy: 0.9104539155960083
# Epoch 77 Accuracy: 0.9051433801651001
# Epoch 78 Accuracy: 0.9109591245651245
# Epoch 79 Accuracy: 0.912513792514801
# Epoch 80 Accuracy: 0.9095758199691772
# Epoch 81 Accuracy: 0.9121890068054199
# Epoch 82 Accuracy: 0.914769172668457
# Epoch 83 Accuracy: 0.9152472615242004
# Epoch 84 Accuracy: 0.9206300377845764
# Epoch 85 Accuracy: 0.922945499420166
# Epoch 86 Accuracy: 0.9240190386772156
# Epoch 87 Accuracy: 0.9235469102859497
# Epoch 88 Accuracy: 0.9300423264503479
# Epoch 89 Accuracy: 0.9331156015396118
# Epoch 90 Accuracy: 0.92670738697052
# Epoch 91 Accuracy: 0.9275373816490173
# Epoch 92 Accuracy: 0.927967369556427
# Epoch 93 Accuracy: 0.9243438243865967
# Epoch 94 Accuracy: 0.9363933205604553
# Epoch 95 Accuracy: 0.9396019577980042
# Epoch 96 Accuracy: 0.928917646408081
# Epoch 97 Accuracy: 0.937554121017456
# Epoch 98 Accuracy: 0.9454237222671509
# Epoch 99 Accuracy: 0.940365731716156
# Epoch 100 Accuracy: 0.943863034248352
# Epoch 101 Accuracy: 0.9416888952255249
# Epoch 102 Accuracy: 0.9402064085006714
# Epoch 103 Accuracy: 0.938080370426178
# Epoch 104 Accuracy: 0.9429007768630981
# Epoch 105 Accuracy: 0.94890296459198
# Epoch 106 Accuracy: 0.9464431405067444
# Epoch 107 Accuracy: 0.9383088946342468
# Epoch 108 Accuracy: 0.9463499188423157
# Epoch 109 Accuracy: 0.9509328007698059
# Epoch 110 Accuracy: 0.9498231410980225
# Epoch 111 Accuracy: 0.945847749710083
# Epoch 112 Accuracy: 0.9522679448127747
# Epoch 113 Accuracy: 0.9472400546073914
# Epoch 114 Accuracy: 0.9502922892570496
# Epoch 115 Accuracy: 0.9529565572738647
# Epoch 116 Accuracy: 0.9542827010154724
# Epoch 117 Accuracy: 0.9520935416221619
# Epoch 118 Accuracy: 0.947008490562439
# Epoch 119 Accuracy: 0.942503809928894
# Epoch 120 Accuracy: 0.9493660926818848
# Epoch 121 Accuracy: 0.9550074338912964
# Epoch 122 Accuracy: 0.9533535242080688
# Epoch 123 Accuracy: 0.9527009725570679
# Epoch 124 Accuracy: 0.9538857936859131
# Epoch 125 Accuracy: 0.954078197479248
# Epoch 126 Accuracy: 0.9480880498886108
# Epoch 127 Accuracy: 0.9465784430503845
# Epoch 128 Accuracy: 0.9576957821846008
# Epoch 129 Accuracy: 0.9571033716201782
# Epoch 130 Accuracy: 0.9544932246208191
# Epoch 131 Accuracy: 0.9604864120483398
# Epoch 132 Accuracy: 0.9590008854866028
# Epoch 133 Accuracy: 0.9570823311805725
# Epoch 134 Accuracy: 0.9626695513725281
# Epoch 135 Accuracy: 0.957467257976532
# Epoch 136 Accuracy: 0.9603781700134277
# Epoch 137 Accuracy: 0.9622846841812134
# Epoch 138 Accuracy: 0.9616771936416626
# Epoch 139 Accuracy: 0.9614035487174988
# Epoch 140 Accuracy: 0.9611389636993408
# Epoch 141 Accuracy: 0.9593587517738342
# Epoch 142 Accuracy: 0.9598669409751892
# Epoch 143 Accuracy: 0.9615899920463562
# Epoch 144 Accuracy: 0.9674929976463318
# Epoch 145 Accuracy: 0.967029869556427
# Epoch 146 Accuracy: 0.9610126614570618
# Epoch 147 Accuracy: 0.9619057774543762
# Epoch 148 Accuracy: 0.9636048078536987
# Epoch 149 Accuracy: 0.966539740562439
# LSTMConfig1 Accuracy: 0.8314711451530457
# Epoch 0 Accuracy: 0.01137593761086464
# Epoch 1 Accuracy: 0.010966969653964043
# Epoch 2 Accuracy: 0.12674713134765625
# Epoch 3 Accuracy: 0.19247376918792725
# Epoch 4 Accuracy: 0.20213264226913452
# Epoch 5 Accuracy: 0.1795070618391037
# Epoch 6 Accuracy: 0.16043290495872498
# Epoch 7 Accuracy: 0.21960702538490295
# Epoch 8 Accuracy: 0.26284340023994446
# Epoch 9 Accuracy: 0.21294625103473663
# Epoch 10 Accuracy: 0.3054242432117462
# Epoch 11 Accuracy: 0.35776618123054504
# Epoch 12 Accuracy: 0.24519461393356323
# Epoch 13 Accuracy: 0.3526179790496826
# Epoch 14 Accuracy: 0.4476580321788788
# Epoch 15 Accuracy: 0.4373105466365814
# Epoch 16 Accuracy: 0.432538241147995
# Epoch 17 Accuracy: 0.4552029073238373
# Epoch 18 Accuracy: 0.4567726254463196
# Epoch 19 Accuracy: 0.45362114906311035
# Epoch 20 Accuracy: 0.4520995616912842
# Epoch 21 Accuracy: 0.4551066756248474
# Epoch 22 Accuracy: 0.45327234268188477
# Epoch 23 Accuracy: 0.45463454723358154
# Epoch 24 Accuracy: 0.4610247015953064
# Epoch 25 Accuracy: 0.4542316198348999
# Epoch 26 Accuracy: 0.45294755697250366
# Epoch 27 Accuracy: 0.4522378742694855
# Epoch 28 Accuracy: 0.45206648111343384
# Epoch 29 Accuracy: 0.4629973769187927
# Epoch 30 Accuracy: 0.4577138423919678
# Epoch 31 Accuracy: 0.4598729610443115
# Epoch 32 Accuracy: 0.45420753955841064
# Epoch 33 Accuracy: 0.4659443497657776
# Epoch 34 Accuracy: 0.45530515909194946
# Epoch 35 Accuracy: 0.4748935401439667
# Epoch 36 Accuracy: 0.4636799693107605
# Epoch 37 Accuracy: 0.47749167680740356
# Epoch 38 Accuracy: 0.46824178099632263
# Epoch 39 Accuracy: 0.4785171151161194
# Epoch 40 Accuracy: 0.47399139404296875
# Epoch 41 Accuracy: 0.4675200581550598
# Epoch 42 Accuracy: 0.46515345573425293
# Epoch 43 Accuracy: 0.472484827041626
# Epoch 44 Accuracy: 0.4799906015396118
# Epoch 45 Accuracy: 0.45889565348625183
# Epoch 46 Accuracy: 0.4641851782798767
# Epoch 47 Accuracy: 0.45854079723358154
# Epoch 48 Accuracy: 0.46405285596847534
# Epoch 49 Accuracy: 0.45598775148391724
# Epoch 50 Accuracy: 0.4472219944000244
# Epoch 51 Accuracy: 0.501458466053009
# Epoch 52 Accuracy: 0.49172738194465637
# Epoch 53 Accuracy: 0.5024117231369019
# Epoch 54 Accuracy: 0.4848170280456543
# Epoch 55 Accuracy: 0.4019317626953125
# Epoch 56 Accuracy: 0.28332188725471497
# Epoch 57 Accuracy: 0.2708423435688019
# Epoch 58 Accuracy: 0.3011420965194702
# Epoch 59 Accuracy: 0.3933825194835663
# Epoch 60 Accuracy: 0.3649832606315613
# Epoch 61 Accuracy: 0.3606500029563904
# Epoch 62 Accuracy: 0.34869369864463806
# Epoch 63 Accuracy: 0.4238506555557251
# Epoch 64 Accuracy: 0.419487327337265
# Epoch 65 Accuracy: 0.4016220271587372
# Epoch 66 Accuracy: 0.39798039197921753
# Epoch 67 Accuracy: 0.4059252142906189
# Epoch 68 Accuracy: 0.3585059344768524
# Epoch 69 Accuracy: 0.35907426476478577
# Epoch 70 Accuracy: 0.3927299678325653
# Epoch 71 Accuracy: 0.3967895805835724
# Epoch 72 Accuracy: 0.3943297564983368
# Epoch 73 Accuracy: 0.41287466883659363
# Epoch 74 Accuracy: 0.3997094929218292
# Epoch 75 Accuracy: 0.3995320796966553
# Epoch 76 Accuracy: 0.42380857467651367
# Epoch 77 Accuracy: 0.43341031670570374
# Epoch 78 Accuracy: 0.4383540153503418
# Epoch 79 Accuracy: 0.4313534200191498
# Epoch 80 Accuracy: 0.4203353524208069
# Epoch 81 Accuracy: 0.33972644805908203
# Epoch 82 Accuracy: 0.40846022963523865
# Epoch 83 Accuracy: 0.42316505312919617
# Epoch 84 Accuracy: 0.42227494716644287
# Epoch 85 Accuracy: 0.4238777160644531
# Epoch 86 Accuracy: 0.3754720985889435
# Epoch 87 Accuracy: 0.29914236068725586
# Epoch 88 Accuracy: 0.33162829279899597
# Epoch 89 Accuracy: 0.4808867275714874
# Epoch 90 Accuracy: 0.45873627066612244
# Epoch 91 Accuracy: 0.4096420109272003
# Epoch 92 Accuracy: 0.40451788902282715
# Epoch 93 Accuracy: 0.40529972314834595
# Epoch 94 Accuracy: 0.42323118448257446
# Epoch 95 Accuracy: 0.4290650188922882
# Epoch 96 Accuracy: 0.4206751585006714
# Epoch 97 Accuracy: 0.4051433801651001
# Epoch 98 Accuracy: 0.3786385953426361
# Epoch 99 Accuracy: 0.39789921045303345
# Epoch 100 Accuracy: 0.43461915850639343
# Epoch 101 Accuracy: 0.40033799409866333
# Epoch 102 Accuracy: 0.3917255997657776
# Epoch 103 Accuracy: 0.4391569197177887
# Epoch 104 Accuracy: 0.5122660398483276
# Epoch 105 Accuracy: 0.5258913040161133
# Epoch 106 Accuracy: 0.48381567001342773
# Epoch 107 Accuracy: 0.41896408796310425
# Epoch 108 Accuracy: 0.4406965672969818
# Epoch 109 Accuracy: 0.5179073810577393
# Epoch 110 Accuracy: 0.5204423666000366
# Epoch 111 Accuracy: 0.49038320779800415
# Epoch 112 Accuracy: 0.4337019920349121
# Epoch 113 Accuracy: 0.43001827597618103
# Epoch 114 Accuracy: 0.42417243123054504
# Epoch 115 Accuracy: 0.417012482881546
# Epoch 116 Accuracy: 0.4356325566768646
# Epoch 117 Accuracy: 0.43562954664230347
# Epoch 118 Accuracy: 0.4272126257419586
# Epoch 119 Accuracy: 0.41638097167015076
# Epoch 120 Accuracy: 0.43997183442115784
# Epoch 121 Accuracy: 0.4424647390842438
# Epoch 122 Accuracy: 0.4486564099788666
# Epoch 123 Accuracy: 0.4462597370147705
# Epoch 124 Accuracy: 0.4307880997657776
# Epoch 125 Accuracy: 0.44379690289497375
# Epoch 126 Accuracy: 0.4352596700191498
# Epoch 127 Accuracy: 0.4318526089191437
# Epoch 128 Accuracy: 0.4354401230812073
# Epoch 129 Accuracy: 0.4377826452255249
# Epoch 130 Accuracy: 0.42462649941444397
# Epoch 131 Accuracy: 0.435560405254364
# Epoch 132 Accuracy: 0.42291244864463806
# Epoch 133 Accuracy: 0.41977301239967346
# Epoch 134 Accuracy: 0.42898380756378174
# Epoch 135 Accuracy: 0.4292183816432953
# Epoch 136 Accuracy: 0.4187716245651245
# Epoch 137 Accuracy: 0.4228883981704712
# Epoch 138 Accuracy: 0.4251317083835602
# Epoch 139 Accuracy: 0.4118281900882721
# Epoch 140 Accuracy: 0.42540836334228516
# Epoch 141 Accuracy: 0.42541736364364624
# Epoch 142 Accuracy: 0.42233806848526
# Epoch 143 Accuracy: 0.42466259002685547
# Epoch 144 Accuracy: 0.42085257172584534
# Epoch 145 Accuracy: 0.42243731021881104
# Epoch 146 Accuracy: 0.4204646348953247
# Epoch 147 Accuracy: 0.4068273603916168
# Epoch 148 Accuracy: 0.41666364669799805
# Epoch 149 Accuracy: 0.41539162397384644
# LSTMConfig2 Accuracy: 0.4746686518192291
# Epoch 0 Accuracy: 0.022947339341044426
# Epoch 1 Accuracy: 0.0015907669439911842
# Epoch 2 Accuracy: 0.0011216560378670692
# Epoch 3 Accuracy: 0.0007698229164816439
# Epoch 4 Accuracy: 0.0007006591185927391
# Epoch 5 Accuracy: 0.0006555523141287267
# Epoch 6 Accuracy: 0.0047602723352611065
# Epoch 7 Accuracy: 0.004396410658955574
# Epoch 8 Accuracy: 0.007277232129126787
# Epoch 9 Accuracy: 0.007271218113601208
# Epoch 10 Accuracy: 0.007821520790457726
# Epoch 11 Accuracy: 0.007060719653964043
# Epoch 12 Accuracy: 0.008898070082068443
# Epoch 13 Accuracy: 0.008353781886398792
# Epoch 14 Accuracy: 0.007761378772556782
# Epoch 15 Accuracy: 0.04339275136590004
# Epoch 16 Accuracy: 0.0665295422077179
# Epoch 17 Accuracy: 0.09101953357458115
# Epoch 18 Accuracy: 0.039916522800922394
# Epoch 19 Accuracy: 0.0744502991437912
# Epoch 20 Accuracy: 0.13444536924362183
# Epoch 21 Accuracy: 0.18542508780956268
# Epoch 22 Accuracy: 0.21540607511997223
# Epoch 23 Accuracy: 0.20642079412937164
# Epoch 24 Accuracy: 0.22159473598003387
# Epoch 25 Accuracy: 0.2370663583278656
# Epoch 26 Accuracy: 0.21805836260318756
# Epoch 27 Accuracy: 0.2198205292224884
# Epoch 28 Accuracy: 0.1856836974620819
# Epoch 29 Accuracy: 0.2107059508562088
# Epoch 30 Accuracy: 0.1863633096218109
# Epoch 31 Accuracy: 0.21948973834514618
# Epoch 32 Accuracy: 0.20974968373775482
# Epoch 33 Accuracy: 0.226664736866951
# Epoch 34 Accuracy: 0.22551901638507843
# Epoch 35 Accuracy: 0.2132950723171234
# Epoch 36 Accuracy: 0.22711580991744995
# Epoch 37 Accuracy: 0.2280750721693039
# Epoch 38 Accuracy: 0.2338758111000061
# Epoch 39 Accuracy: 0.22892609238624573
# Epoch 40 Accuracy: 0.2413424849510193
# Epoch 41 Accuracy: 0.19342702627182007
# Epoch 42 Accuracy: 0.17977169156074524
# Epoch 43 Accuracy: 0.19943825900554657
# Epoch 44 Accuracy: 0.212023064494133
# Epoch 45 Accuracy: 0.19453966617584229
# Epoch 46 Accuracy: 0.20461051166057587
# Epoch 47 Accuracy: 0.21190878748893738
# Epoch 48 Accuracy: 0.22303815186023712
# Epoch 49 Accuracy: 0.23459450900554657
# Epoch 50 Accuracy: 0.23359614610671997
# Epoch 51 Accuracy: 0.2297530472278595
# Epoch 52 Accuracy: 0.22461086511611938
# Epoch 53 Accuracy: 0.20361515879631042
# Epoch 54 Accuracy: 0.21524669229984283
# Epoch 55 Accuracy: 0.23113933205604553
# Epoch 56 Accuracy: 0.22054524719715118
# Epoch 57 Accuracy: 0.22864341735839844
# Epoch 58 Accuracy: 0.22548894584178925
# Epoch 59 Accuracy: 0.22699551284313202
# Epoch 60 Accuracy: 0.22763904929161072
# Epoch 61 Accuracy: 0.2290794551372528
# Epoch 62 Accuracy: 0.23541244864463806
# Epoch 63 Accuracy: 0.2410959005355835
# Epoch 64 Accuracy: 0.23978179693222046
# Epoch 65 Accuracy: 0.24526077508926392
# Epoch 66 Accuracy: 0.2452036291360855
# Epoch 67 Accuracy: 0.24783185124397278
# Epoch 68 Accuracy: 0.24877609312534332
# Epoch 69 Accuracy: 0.2366032749414444
# Epoch 70 Accuracy: 0.2295936644077301
# Epoch 71 Accuracy: 0.10340586304664612
# Epoch 72 Accuracy: 0.2857516407966614
# Epoch 73 Accuracy: 0.36797234416007996
# Epoch 74 Accuracy: 0.3679633140563965
# Epoch 75 Accuracy: 0.3988073766231537
# Epoch 76 Accuracy: 0.41359037160873413
# Epoch 77 Accuracy: 0.42069318890571594
# Epoch 78 Accuracy: 0.41338589787483215
# Epoch 79 Accuracy: 0.40608760714530945
# Epoch 80 Accuracy: 0.4200797379016876
# Epoch 81 Accuracy: 0.4112568497657776
# Epoch 82 Accuracy: 0.4042171835899353
# Epoch 83 Accuracy: 0.4036397933959961
# Epoch 84 Accuracy: 0.40492984652519226
# Epoch 85 Accuracy: 0.40613269805908203
# Epoch 86 Accuracy: 0.4214569926261902
# Epoch 87 Accuracy: 0.4313865005970001
# Epoch 88 Accuracy: 0.4264458119869232
# Epoch 89 Accuracy: 0.4240040183067322
# Epoch 90 Accuracy: 0.4285447895526886
# Epoch 91 Accuracy: 0.4123333990573883
# Epoch 92 Accuracy: 0.41284459829330444
# Epoch 93 Accuracy: 0.4212464988231659
# Epoch 94 Accuracy: 0.43574684858322144
# Epoch 95 Accuracy: 0.4338192641735077
# Epoch 96 Accuracy: 0.4518108665943146
# Epoch 97 Accuracy: 0.4646632969379425
# Epoch 98 Accuracy: 0.46935442090034485
# Epoch 99 Accuracy: 0.44598910212516785
# Epoch 100 Accuracy: 0.4618065357208252
# Epoch 101 Accuracy: 0.45820099115371704
# Epoch 102 Accuracy: 0.444329172372818
# Epoch 103 Accuracy: 0.4369136095046997
# Epoch 104 Accuracy: 0.4559095799922943
# Epoch 105 Accuracy: 0.44434118270874023
# Epoch 106 Accuracy: 0.4316992461681366
# Epoch 107 Accuracy: 0.4351905286312103
# Epoch 108 Accuracy: 0.4298619031906128
# Epoch 109 Accuracy: 0.4528423249721527
# Epoch 110 Accuracy: 0.45408424735069275
# Epoch 111 Accuracy: 0.4654301106929779
# Epoch 112 Accuracy: 0.468070387840271
# Epoch 113 Accuracy: 0.47537466883659363
# Epoch 114 Accuracy: 0.46053755283355713
# Epoch 115 Accuracy: 0.4624921679496765
# Epoch 116 Accuracy: 0.4671802818775177
# Epoch 117 Accuracy: 0.4504757225513458
# Epoch 118 Accuracy: 0.4827902317047119
# Epoch 119 Accuracy: 0.4870723783969879
# Epoch 120 Accuracy: 0.476418137550354
# Epoch 121 Accuracy: 0.4549623429775238
# Epoch 122 Accuracy: 0.4512665867805481
# Epoch 123 Accuracy: 0.45996618270874023
# Epoch 124 Accuracy: 0.47125792503356934
# Epoch 125 Accuracy: 0.4997624158859253
# Epoch 126 Accuracy: 0.48740315437316895
# Epoch 127 Accuracy: 0.5048835277557373
# Epoch 128 Accuracy: 0.5382535457611084
# Epoch 129 Accuracy: 0.5310094356536865
# Epoch 130 Accuracy: 0.5247636437416077
# Epoch 131 Accuracy: 0.5665144920349121
# Epoch 132 Accuracy: 0.5715604424476624
# Epoch 133 Accuracy: 0.5617151260375977
# Epoch 134 Accuracy: 0.5589154958724976
# Epoch 135 Accuracy: 0.5575442314147949
# Epoch 136 Accuracy: 0.5857871174812317
# Epoch 137 Accuracy: 0.5811501741409302
# Epoch 138 Accuracy: 0.5778934359550476
# Epoch 139 Accuracy: 0.5783054232597351
# Epoch 140 Accuracy: 0.5944656729698181
# Epoch 141 Accuracy: 0.6075105667114258
# Epoch 142 Accuracy: 0.6001340746879578
# Epoch 143 Accuracy: 0.5925140380859375
# Epoch 144 Accuracy: 0.5937018394470215
# Epoch 145 Accuracy: 0.6055529117584229
# Epoch 146 Accuracy: 0.6084488034248352
# Epoch 147 Accuracy: 0.6045876741409302
# Epoch 148 Accuracy: 0.616480827331543
# Epoch 149 Accuracy: 0.622209370136261
# LSTMConfig3 Accuracy: 0.679866373538971
# Epoch 0 Accuracy: 0.048760462552309036
# Epoch 1 Accuracy: 0.07894594222307205
# Epoch 2 Accuracy: 0.13271626830101013
# Epoch 3 Accuracy: 0.1237851157784462
# Epoch 4 Accuracy: 0.16762292385101318
# Epoch 5 Accuracy: 0.27824586629867554
# Epoch 6 Accuracy: 0.3207755982875824
# Epoch 7 Accuracy: 0.30547836422920227
# Epoch 8 Accuracy: 0.3074149489402771
# Epoch 9 Accuracy: 0.3626948595046997
# Epoch 10 Accuracy: 0.2758973240852356
# Epoch 11 Accuracy: 0.31394943594932556
# Epoch 12 Accuracy: 0.3548973798751831
# Epoch 13 Accuracy: 0.5150386095046997
# Epoch 14 Accuracy: 0.5072742104530334
# Epoch 15 Accuracy: 0.4911951422691345
# Epoch 16 Accuracy: 0.529845654964447
# Epoch 17 Accuracy: 0.4851568341255188
# Epoch 18 Accuracy: 0.5134989619255066
# Epoch 19 Accuracy: 0.4561050534248352
# Epoch 20 Accuracy: 0.449047327041626
# Epoch 21 Accuracy: 0.44400739669799805
# Epoch 22 Accuracy: 0.4830217659473419
# Epoch 23 Accuracy: 0.47628283500671387
# Epoch 24 Accuracy: 0.47948840260505676
# Epoch 25 Accuracy: 0.47707971930503845
# Epoch 26 Accuracy: 0.46954986453056335
# Epoch 27 Accuracy: 0.45417746901512146
# Epoch 28 Accuracy: 0.44373375177383423
# Epoch 29 Accuracy: 0.4681786298751831
# Epoch 30 Accuracy: 0.45833632349967957
# Epoch 31 Accuracy: 0.5076831579208374
# Epoch 32 Accuracy: 0.5894137024879456
# Epoch 33 Accuracy: 0.5661326050758362
# Epoch 34 Accuracy: 0.48902398347854614
# Epoch 35 Accuracy: 0.5237592458724976
# Epoch 36 Accuracy: 0.5388068556785583
# Epoch 37 Accuracy: 0.5131441354751587
# Epoch 38 Accuracy: 0.5211580991744995
# Epoch 39 Accuracy: 0.4539218842983246
# Epoch 40 Accuracy: 0.4853312373161316
# Epoch 41 Accuracy: 0.43199095129966736
# Epoch 42 Accuracy: 0.45683276653289795
# Epoch 43 Accuracy: 0.4534587860107422
# Epoch 44 Accuracy: 0.4708279073238373
# Epoch 45 Accuracy: 0.4723013937473297
# Epoch 46 Accuracy: 0.49389252066612244
# Epoch 47 Accuracy: 0.4800357222557068
# Epoch 48 Accuracy: 0.4909304976463318
# Epoch 49 Accuracy: 0.4807874858379364
# Epoch 50 Accuracy: 0.45034340023994446
# Epoch 51 Accuracy: 0.4814460575580597
# Epoch 52 Accuracy: 0.45468267798423767
# Epoch 53 Accuracy: 0.47817128896713257
# Epoch 54 Accuracy: 0.47456276416778564
# Epoch 55 Accuracy: 0.46230873465538025
# Epoch 56 Accuracy: 0.48119646310806274
# Epoch 57 Accuracy: 0.48756253719329834
# Epoch 58 Accuracy: 0.4996812343597412
# Epoch 59 Accuracy: 0.5048925876617432
# Epoch 60 Accuracy: 0.5062668323516846
# Epoch 61 Accuracy: 0.5206137895584106
# Epoch 62 Accuracy: 0.5319265723228455
# Epoch 63 Accuracy: 0.5172037482261658
# Epoch 64 Accuracy: 0.5320979952812195
# Epoch 65 Accuracy: 0.5493679046630859
# Epoch 66 Accuracy: 0.5373935103416443
# Epoch 67 Accuracy: 0.5902316570281982
# Epoch 68 Accuracy: 0.5876124501228333
# Epoch 69 Accuracy: 0.6135669350624084
# Epoch 70 Accuracy: 0.6202006340026855
# Epoch 71 Accuracy: 0.6281123757362366
# Epoch 72 Accuracy: 0.6275981068611145
# Epoch 73 Accuracy: 0.6288852095603943
# Epoch 74 Accuracy: 0.6422999501228333
# Epoch 75 Accuracy: 0.6558981537818909
# Epoch 76 Accuracy: 0.6563191413879395
# Epoch 77 Accuracy: 0.6707954406738281
# Epoch 78 Accuracy: 0.6843094229698181
# Epoch 79 Accuracy: 0.6786049008369446
# Epoch 80 Accuracy: 0.6975497603416443
# Epoch 81 Accuracy: 0.7018679976463318
# Epoch 82 Accuracy: 0.7083964347839355
# Epoch 83 Accuracy: 0.7228577136993408
# Epoch 84 Accuracy: 0.7298522591590881
# Epoch 85 Accuracy: 0.7426716089248657
# Epoch 86 Accuracy: 0.752474844455719
# Epoch 87 Accuracy: 0.7627291083335876
# Epoch 88 Accuracy: 0.771735429763794
# Epoch 89 Accuracy: 0.7772535085678101
# Epoch 90 Accuracy: 0.7820769548416138
# Epoch 91 Accuracy: 0.7975846529006958
# Epoch 92 Accuracy: 0.8101935386657715
# Epoch 93 Accuracy: 0.8147974014282227
# Epoch 94 Accuracy: 0.825532853603363
# Epoch 95 Accuracy: 0.8346203565597534
# Epoch 96 Accuracy: 0.837777853012085
# Epoch 97 Accuracy: 0.8414344787597656
# Epoch 98 Accuracy: 0.8466699123382568
# Epoch 99 Accuracy: 0.8541936874389648
# Epoch 100 Accuracy: 0.8612333536148071
# Epoch 101 Accuracy: 0.8662673234939575
# Epoch 102 Accuracy: 0.8701103925704956
# Epoch 103 Accuracy: 0.8721402287483215
# Epoch 104 Accuracy: 0.8767982125282288
# Epoch 105 Accuracy: 0.8808548450469971
# Epoch 106 Accuracy: 0.8833386898040771
# Epoch 107 Accuracy: 0.8867096900939941
# Epoch 108 Accuracy: 0.890435516834259
# Epoch 109 Accuracy: 0.8928953409194946
# Epoch 110 Accuracy: 0.894750714302063
# Epoch 111 Accuracy: 0.8969249129295349
# Epoch 112 Accuracy: 0.8995831608772278
# Epoch 113 Accuracy: 0.9034202694892883
# Epoch 114 Accuracy: 0.90413898229599
# Epoch 115 Accuracy: 0.9062108993530273
# Epoch 116 Accuracy: 0.9070318341255188
# Epoch 117 Accuracy: 0.9097291827201843
# Epoch 118 Accuracy: 0.9112117290496826
# Epoch 119 Accuracy: 0.9139091372489929
# Epoch 120 Accuracy: 0.9161764979362488
# Epoch 121 Accuracy: 0.9172950983047485
# Epoch 122 Accuracy: 0.9192798137664795
# Epoch 123 Accuracy: 0.9220253229141235
# Epoch 124 Accuracy: 0.9220012426376343
# Epoch 125 Accuracy: 0.9246355295181274
# Epoch 126 Accuracy: 0.9273028373718262
# Epoch 127 Accuracy: 0.9282981753349304
# Epoch 128 Accuracy: 0.9289266467094421
# Epoch 129 Accuracy: 0.9284515380859375
# Epoch 130 Accuracy: 0.9310888051986694
# Epoch 131 Accuracy: 0.9329892992973328
# Epoch 132 Accuracy: 0.9348927736282349
# Epoch 133 Accuracy: 0.9361557960510254
# Epoch 134 Accuracy: 0.9357377886772156
# Epoch 135 Accuracy: 0.9368654489517212
# Epoch 136 Accuracy: 0.9375962018966675
# Epoch 137 Accuracy: 0.9391177892684937
# Epoch 138 Accuracy: 0.9403266906738281
# Epoch 139 Accuracy: 0.9416678547859192
# Epoch 140 Accuracy: 0.9421790242195129
# Epoch 141 Accuracy: 0.9415535926818848
# Epoch 142 Accuracy: 0.9427503943443298
# Epoch 143 Accuracy: 0.9443652033805847
# Epoch 144 Accuracy: 0.9437127113342285
# Epoch 145 Accuracy: 0.9442840218544006
# Epoch 146 Accuracy: 0.9453004598617554
# Epoch 147 Accuracy: 0.9459680318832397
# Epoch 148 Accuracy: 0.9478805661201477
# Epoch 149 Accuracy: 0.9457395076751709
# LSTMConfig4 Accuracy: 0.8746029138565063
# Epoch 0 Accuracy: 0.0844549834728241
# Epoch 1 Accuracy: 0.1973092257976532
# Epoch 2 Accuracy: 0.22294794023036957
# Epoch 3 Accuracy: 0.2340863049030304
# Epoch 4 Accuracy: 0.2502104938030243
# Epoch 5 Accuracy: 0.2550429403781891
# Epoch 6 Accuracy: 0.25534966588020325
# Epoch 7 Accuracy: 0.2524026930332184
# Epoch 8 Accuracy: 0.24279192090034485
# Epoch 9 Accuracy: 0.23432688415050507
# Epoch 10 Accuracy: 0.230294331908226
# Epoch 11 Accuracy: 0.22647227346897125
# Epoch 12 Accuracy: 0.22303514182567596
# Epoch 13 Accuracy: 0.21440470218658447
# Epoch 14 Accuracy: 0.2134123593568802
# Epoch 15 Accuracy: 0.2124861627817154
# Epoch 16 Accuracy: 0.21345746517181396
# Epoch 17 Accuracy: 0.21468135714530945
# Epoch 18 Accuracy: 0.21330410242080688
# Epoch 19 Accuracy: 0.21053153276443481
# Epoch 20 Accuracy: 0.2091302126646042
# Epoch 21 Accuracy: 0.21399573981761932
# Epoch 22 Accuracy: 0.2176433652639389
# Epoch 23 Accuracy: 0.21377621591091156
# Epoch 24 Accuracy: 0.21397769451141357
# Epoch 25 Accuracy: 0.21901161968708038
# Epoch 26 Accuracy: 0.22485744953155518
# Epoch 27 Accuracy: 0.2254408299922943
# Epoch 28 Accuracy: 0.22512207925319672
# Epoch 29 Accuracy: 0.23134681582450867
# Epoch 30 Accuracy: 0.23401112854480743
# Epoch 31 Accuracy: 0.2357071489095688
# Epoch 32 Accuracy: 0.2363205999135971
# Epoch 33 Accuracy: 0.23996523022651672
# Epoch 34 Accuracy: 0.2504690885543823
# Epoch 35 Accuracy: 0.24695979058742523
# Epoch 36 Accuracy: 0.24387750029563904
# Epoch 37 Accuracy: 0.24806340038776398
# Epoch 38 Accuracy: 0.25229743123054504
# Epoch 39 Accuracy: 0.25934913754463196
# Epoch 40 Accuracy: 0.265276163816452
# Epoch 41 Accuracy: 0.26157140731811523
# Epoch 42 Accuracy: 0.26625046133995056
# Epoch 43 Accuracy: 0.2683825194835663
# Epoch 44 Accuracy: 0.2751876413822174
# Epoch 45 Accuracy: 0.27712422609329224
# Epoch 46 Accuracy: 0.28102144598960876
# Epoch 47 Accuracy: 0.2846420109272003
# Epoch 48 Accuracy: 0.29449033737182617
# Epoch 49 Accuracy: 0.29009994864463806
# Epoch 50 Accuracy: 0.2983815670013428
# Epoch 51 Accuracy: 0.3076765537261963
# Epoch 52 Accuracy: 0.30780887603759766
# Epoch 53 Accuracy: 0.3133810758590698
# Epoch 54 Accuracy: 0.31862249970436096
# Epoch 55 Accuracy: 0.32718977332115173
# Epoch 56 Accuracy: 0.3240323066711426
# Epoch 57 Accuracy: 0.33330926299095154
# Epoch 58 Accuracy: 0.33667123317718506
# Epoch 59 Accuracy: 0.3365870416164398
# Epoch 60 Accuracy: 0.34104058146476746
# Epoch 61 Accuracy: 0.3486846685409546
# Epoch 62 Accuracy: 0.35025739669799805
# Epoch 63 Accuracy: 0.36282116174697876
# Epoch 64 Accuracy: 0.3583255112171173
# Epoch 65 Accuracy: 0.36274296045303345
# Epoch 66 Accuracy: 0.3731265664100647
# Epoch 67 Accuracy: 0.3697856366634369
# Epoch 68 Accuracy: 0.3759622573852539
# Epoch 69 Accuracy: 0.37132528424263
# Epoch 70 Accuracy: 0.3803376257419586
# Epoch 71 Accuracy: 0.39251646399497986
# Epoch 72 Accuracy: 0.39344266057014465
# Epoch 73 Accuracy: 0.3946545422077179
# Epoch 74 Accuracy: 0.39587241411209106
# Epoch 75 Accuracy: 0.4007589817047119
# Epoch 76 Accuracy: 0.3947206735610962
# Epoch 77 Accuracy: 0.39333438873291016
# Epoch 78 Accuracy: 0.3912203907966614
# Epoch 79 Accuracy: 0.3862646520137787
# Epoch 80 Accuracy: 0.3948409855365753
# Epoch 81 Accuracy: 0.3977729082107544
# Epoch 82 Accuracy: 0.3978360593318939
# Epoch 83 Accuracy: 0.40780165791511536
# Epoch 84 Accuracy: 0.4054110050201416
# Epoch 85 Accuracy: 0.4158487021923065
# Epoch 86 Accuracy: 0.4076603353023529
# Epoch 87 Accuracy: 0.40924808382987976
# Epoch 88 Accuracy: 0.41437220573425293
# Epoch 89 Accuracy: 0.4123634696006775
# Epoch 90 Accuracy: 0.41571640968322754
# Epoch 91 Accuracy: 0.41845589876174927
# Epoch 92 Accuracy: 0.4225756525993347
# Epoch 93 Accuracy: 0.42868611216545105
# Epoch 94 Accuracy: 0.4252970814704895
# Epoch 95 Accuracy: 0.4301295280456543
# Epoch 96 Accuracy: 0.42982882261276245
# Epoch 97 Accuracy: 0.44031164050102234
# Epoch 98 Accuracy: 0.4426572024822235
# Epoch 99 Accuracy: 0.4456462860107422
# Epoch 100 Accuracy: 0.4550495445728302
# Epoch 101 Accuracy: 0.4517236649990082
# Epoch 102 Accuracy: 0.4431924521923065
# Epoch 103 Accuracy: 0.45255663990974426
# Epoch 104 Accuracy: 0.46003836393356323
# Epoch 105 Accuracy: 0.458450585603714
# Epoch 106 Accuracy: 0.46401676535606384
# Epoch 107 Accuracy: 0.4609314799308777
# Epoch 108 Accuracy: 0.45278817415237427
# Epoch 109 Accuracy: 0.45530515909194946
# Epoch 110 Accuracy: 0.4538196325302124
# Epoch 111 Accuracy: 0.4640498459339142
# Epoch 112 Accuracy: 0.4569229781627655
# Epoch 113 Accuracy: 0.46356871724128723
# Epoch 114 Accuracy: 0.4768150746822357
# Epoch 115 Accuracy: 0.46909579634666443
# Epoch 116 Accuracy: 0.465003103017807
# Epoch 117 Accuracy: 0.46057963371276855
# Epoch 118 Accuracy: 0.465003103017807
# Epoch 119 Accuracy: 0.4662240147590637
# Epoch 120 Accuracy: 0.4507463574409485
# Epoch 121 Accuracy: 0.4510350227355957
# Epoch 122 Accuracy: 0.46604058146476746
# Epoch 123 Accuracy: 0.4784599840641022
# Epoch 124 Accuracy: 0.475203275680542
# Epoch 125 Accuracy: 0.47104743123054504
# Epoch 126 Accuracy: 0.47031670808792114
# Epoch 127 Accuracy: 0.48851278424263
# Epoch 128 Accuracy: 0.4920942783355713
# Epoch 129 Accuracy: 0.47379592061042786
# Epoch 130 Accuracy: 0.4822339117527008
# Epoch 131 Accuracy: 0.49441877007484436
# Epoch 132 Accuracy: 0.4929693341255188
# Epoch 133 Accuracy: 0.4968394935131073
# Epoch 134 Accuracy: 0.4954562187194824
# Epoch 135 Accuracy: 0.4880135953426361
# Epoch 136 Accuracy: 0.5037708878517151
# Epoch 137 Accuracy: 0.5182742476463318
# Epoch 138 Accuracy: 0.48984494805336
# Epoch 139 Accuracy: 0.4921092987060547
# Epoch 140 Accuracy: 0.5078094601631165
# Epoch 141 Accuracy: 0.5181660056114197
# Epoch 142 Accuracy: 0.5248027443885803
# Epoch 143 Accuracy: 0.526393473148346
# Epoch 144 Accuracy: 0.5266701579093933
# Epoch 145 Accuracy: 0.5251274704933167
# Epoch 146 Accuracy: 0.5108526945114136
# Epoch 147 Accuracy: 0.507523775100708
# Epoch 148 Accuracy: 0.5049827694892883
# Epoch 149 Accuracy: 0.508320689201355
# LSTMConfig5 Accuracy: 0.3187370002269745
# Epoch 0 Accuracy: 0.07693418115377426
# Epoch 1 Accuracy: 0.16743949055671692
# Epoch 2 Accuracy: 0.2130424827337265
# Epoch 3 Accuracy: 0.24400679767131805
# Epoch 4 Accuracy: 0.2550489604473114
# Epoch 5 Accuracy: 0.25748470425605774
# Epoch 6 Accuracy: 0.25972801446914673
# Epoch 7 Accuracy: 0.2577102482318878
# Epoch 8 Accuracy: 0.25588491559028625
# Epoch 9 Accuracy: 0.25113970041275024
# Epoch 10 Accuracy: 0.24486082792282104
# Epoch 11 Accuracy: 0.253734827041626
# Epoch 12 Accuracy: 0.25351831316947937
# Epoch 13 Accuracy: 0.2578425705432892
# Epoch 14 Accuracy: 0.261995404958725
# Epoch 15 Accuracy: 0.26238030195236206
# Epoch 16 Accuracy: 0.26952221989631653
# Epoch 17 Accuracy: 0.26749542355537415
# Epoch 18 Accuracy: 0.2755936086177826
# Epoch 19 Accuracy: 0.2856103181838989
# Epoch 20 Accuracy: 0.2879648804664612
# Epoch 21 Accuracy: 0.2920876443386078
# Epoch 22 Accuracy: 0.3000865876674652
# Epoch 23 Accuracy: 0.3056076765060425
# Epoch 24 Accuracy: 0.3106686472892761
# Epoch 25 Accuracy: 0.31832778453826904
# Epoch 26 Accuracy: 0.32382479310035706
# Epoch 27 Accuracy: 0.33000144362449646
# Epoch 28 Accuracy: 0.33895063400268555
# Epoch 29 Accuracy: 0.3465797007083893
# Epoch 30 Accuracy: 0.3517128527164459
# Epoch 31 Accuracy: 0.3570655286312103
# Epoch 32 Accuracy: 0.366204172372818
# Epoch 33 Accuracy: 0.3723747730255127
# Epoch 34 Accuracy: 0.37811535596847534
# Epoch 35 Accuracy: 0.38570234179496765
# Epoch 36 Accuracy: 0.3969910740852356
# Epoch 37 Accuracy: 0.40654468536376953
# Epoch 38 Accuracy: 0.4065777659416199
# Epoch 39 Accuracy: 0.4187987148761749
# Epoch 40 Accuracy: 0.42373037338256836
# Epoch 41 Accuracy: 0.4287252128124237
# Epoch 42 Accuracy: 0.4408559203147888
# Epoch 43 Accuracy: 0.44401341676712036
# Epoch 44 Accuracy: 0.45353394746780396
# Epoch 45 Accuracy: 0.4554615318775177
# Epoch 46 Accuracy: 0.4565380811691284
# Epoch 47 Accuracy: 0.4664856195449829
# Epoch 48 Accuracy: 0.47914260625839233
# Epoch 49 Accuracy: 0.4817948639392853
# Epoch 50 Accuracy: 0.476186603307724
# Epoch 51 Accuracy: 0.4857432246208191
# Epoch 52 Accuracy: 0.49429547786712646
# Epoch 53 Accuracy: 0.5022733807563782
# Epoch 54 Accuracy: 0.515579879283905
# Epoch 55 Accuracy: 0.5153122544288635
# Epoch 56 Accuracy: 0.5153483152389526
# Epoch 57 Accuracy: 0.515625
# Epoch 58 Accuracy: 0.5204965472221375
# Epoch 59 Accuracy: 0.525304913520813
# Epoch 60 Accuracy: 0.5369154214859009
# Epoch 61 Accuracy: 0.5420034527778625
# Epoch 62 Accuracy: 0.536244809627533
# Epoch 63 Accuracy: 0.5364913940429688
# Epoch 64 Accuracy: 0.5496295094490051
# Epoch 65 Accuracy: 0.5575171709060669
# Epoch 66 Accuracy: 0.5509856939315796
# Epoch 67 Accuracy: 0.5393301248550415
# Epoch 68 Accuracy: 0.5501797795295715
# Epoch 69 Accuracy: 0.5634893178939819
# Epoch 70 Accuracy: 0.5692870616912842
# Epoch 71 Accuracy: 0.5737676620483398
# Epoch 72 Accuracy: 0.5637509226799011
# Epoch 73 Accuracy: 0.5684661269187927
# Epoch 74 Accuracy: 0.5817245244979858
# Epoch 75 Accuracy: 0.5782843828201294
# Epoch 76 Accuracy: 0.5756891965866089
# Epoch 77 Accuracy: 0.5845031142234802
# Epoch 78 Accuracy: 0.5751148462295532
# Epoch 79 Accuracy: 0.5806058645248413
# Epoch 80 Accuracy: 0.5926012396812439
# Epoch 81 Accuracy: 0.5926343202590942
# Epoch 82 Accuracy: 0.5906616449356079
# Epoch 83 Accuracy: 0.5906887054443359
# Epoch 84 Accuracy: 0.594158947467804
# Epoch 85 Accuracy: 0.5961917638778687
# Epoch 86 Accuracy: 0.6101598143577576
# Epoch 87 Accuracy: 0.5998243689537048
# Epoch 88 Accuracy: 0.5960835218429565
# Epoch 89 Accuracy: 0.6001611351966858
# Epoch 90 Accuracy: 0.6110469102859497
# Epoch 91 Accuracy: 0.6081660985946655
# Epoch 92 Accuracy: 0.6108545064926147
# Epoch 93 Accuracy: 0.6163274645805359
# Epoch 94 Accuracy: 0.6049003601074219
# Epoch 95 Accuracy: 0.6113356351852417
# Epoch 96 Accuracy: 0.6310533285140991
# Epoch 97 Accuracy: 0.6359909772872925
# Epoch 98 Accuracy: 0.6328575611114502
# Epoch 99 Accuracy: 0.6243414282798767
# Epoch 100 Accuracy: 0.6335011124610901
# Epoch 101 Accuracy: 0.6292821168899536
# Epoch 102 Accuracy: 0.6341235637664795
# Epoch 103 Accuracy: 0.6463565230369568
# Epoch 104 Accuracy: 0.6451025605201721
# Epoch 105 Accuracy: 0.6397649645805359
# Epoch 106 Accuracy: 0.6357865333557129
# Epoch 107 Accuracy: 0.6441704034805298
# Epoch 108 Accuracy: 0.6409557461738586
# Epoch 109 Accuracy: 0.6395995616912842
# Epoch 110 Accuracy: 0.6425796151161194
# Epoch 111 Accuracy: 0.6576151847839355
# Epoch 112 Accuracy: 0.6584932804107666
# Epoch 113 Accuracy: 0.6596239805221558
# Epoch 114 Accuracy: 0.6606012582778931
# Epoch 115 Accuracy: 0.6518625617027283
# Epoch 116 Accuracy: 0.6485878229141235
# Epoch 117 Accuracy: 0.6582587361335754
# Epoch 118 Accuracy: 0.6607065200805664
# Epoch 119 Accuracy: 0.6579309701919556
# Epoch 120 Accuracy: 0.6599758267402649
# Epoch 121 Accuracy: 0.6712765693664551
# Epoch 122 Accuracy: 0.6793386340141296
# Epoch 123 Accuracy: 0.6820089817047119
# Epoch 124 Accuracy: 0.6687836647033691
# Epoch 125 Accuracy: 0.6688317656517029
# Epoch 126 Accuracy: 0.6705819368362427
# Epoch 127 Accuracy: 0.6795190572738647
# Epoch 128 Accuracy: 0.6906153559684753
# Epoch 129 Accuracy: 0.6784725785255432
# Epoch 130 Accuracy: 0.677558422088623
# Epoch 131 Accuracy: 0.6885945796966553
# Epoch 132 Accuracy: 0.6844808459281921
# Epoch 133 Accuracy: 0.6898124814033508
# Epoch 134 Accuracy: 0.6948403716087341
# Epoch 135 Accuracy: 0.7005568742752075
# Epoch 136 Accuracy: 0.6901823282241821
# Epoch 137 Accuracy: 0.6784335374832153
# Epoch 138 Accuracy: 0.6724042296409607
# Epoch 139 Accuracy: 0.6876984238624573
# Epoch 140 Accuracy: 0.6993299722671509
# Epoch 141 Accuracy: 0.6969423294067383
# Epoch 142 Accuracy: 0.6970265507698059
# Epoch 143 Accuracy: 0.7013357281684875
# Epoch 144 Accuracy: 0.7000817656517029
# Epoch 145 Accuracy: 0.6980308890342712
# Epoch 146 Accuracy: 0.6836779117584229
# Epoch 147 Accuracy: 0.6979166269302368
# Epoch 148 Accuracy: 0.7091512680053711
# Epoch 149 Accuracy: 0.718677818775177
# LSTMConfig6 Accuracy: 0.5127889513969421