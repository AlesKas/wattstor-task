import torch

import numpy as np
import pandas as pd
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def absolute_error_statistics(original : np.array, predicted : np.array):
    assert len(original) == len(predicted)
    return np.absolute(original - predicted)

# Class representing the Dataset
class CNNDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        return torch.tensor(item), torch.tensor(label)

class TimeseriesCNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(TimeseriesCNN,self).__init__()
        self.conv1d = nn.Conv1d(3, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNN():
    def __init__(self, input_file, quantity) -> None:
        self.data = pd.read_csv(input_file, delimiter=';')
        self.data['Time'] = pd.to_datetime(self.data['Time'], utc=True)
        self.data.set_index('Time')

        if quantity not in self.data.columns:
            raise Exception(f"{quantity} column not in data.")
        
        self.fill_missing_data()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TimeseriesCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()
        train_len = int(len(self.data[quantity]) * 0.80)

        # Split datasets into the train and test
        self.train_data = self.data[quantity][:train_len]
        self.test_data = self.data[quantity][train_len:]

        n_steps = 3
        train_x, train_y = self.split_sequence(self.train_data.values, n_steps)
        test_x, test_y = self.split_sequence(self.test_data.values, n_steps)

        self.train_dataset = CNNDataset(train_x.reshape(train_x.shape[0],train_x.shape[1], 1), train_y)
        self.test_dataset = CNNDataset(test_x.reshape(test_x.shape[0],test_x.shape[1], 1), test_y)

        self.train_loader = DataLoader(self.train_dataset, batch_size=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1)

    def fill_missing_data(self):
        missing = {}
        columns = [i for i in self.data.columns if i not in ['Time']]
        for column in columns:
            missing[column] = self.data[self.data[column].isna()].index.to_list()

        for key, value in missing.items():
            if len(value) == 0:
                continue
            start_index = value[0] - 1
            end_index = value[-1] + 1

            start_value = self.data[key][start_index]
            end_value = self.data[key][end_index]

            fill_data = float((end_value - start_value) / (len(value) + 1))
            new_value = start_value + fill_data
            for item in value:
                self.data.at[item, key] = new_value
                new_value += fill_data

    # Helper function to crate input values to the model
    # Format: [dato1, dato2, dato3], [predicted_value]
    def split_sequence(self, sequence, n_steps):
        x, y = list(), list()
        for i in range(len(sequence)):    
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train(self):
        running_loss = .0
        self.model.train()

        for idx, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(inputs.float())
            loss = self.criterion(preds,labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss
            
        train_loss = running_loss/len(self.train_loader)
        return train_loss.detach().cpu().numpy()
    
    def evaluate(self):
        running_loss = .0
        self.model.eval()
        predicted = []
        actuals = []
        
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(inputs.float())
                loss = self.criterion(preds,labels)
                running_loss += loss
                predicted.extend(preds.detach().cpu().numpy())
                actuals.extend(labels.detach().cpu().numpy())
                
            test_loss = running_loss/len(self.test_loader)
            return test_loss.detach().cpu().numpy(), predicted, actuals
        