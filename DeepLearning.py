import torch
import torch.nn as nn
import MachineLearning as utils
import pickle
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 


import matplotlib.pyplot as plt


def load_data (list_of_files, batch_size=25):

    df= pd.DataFrame()
    for (file) in list_of_files:
        df_aux = utils.read_parquet('data/'+file) # type: ignore
        df_aux = utils.add_features(df_aux)
        print(df_aux.isna().sum())
        print(df_aux.shape)
        df = pd.concat([df, df_aux], axis=0) # type: ignore

    print("Datasets cargados")
    scaler = pickle.load(open('models/scale_model.pkl', 'rb'))
    cols = pickle.load(open('models/filtered_cols.pkl', 'rb'))
    df2 = pd.DataFrame()
    df2[cols] = scaler.transform(df[cols])
    #pca = pickle.load(open('models/pca_model.pkl', 'rb'))
    #df2 = pca.transform(df2)
    df2["target"] = df["target"].to_list()
    X_train, X_test, y_train, y_test = utils.train_test_split(df2) # type: ignore 

    # Convert X_train, X_test, y_train, y_test to tensors
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    X_train = torch.tensor(X_train.to_numpy() , dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy() , dtype=torch.float32)

    # Crear un dataset y un DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, X_test, y_test

# Definir el modelo, función de pérdida y optimizador
class Deep(nn.Module):
    def __init__(self, shape):
        super(Deep, self).__init__()
        self.layer1 = nn.Linear(shape, 256)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(128, 60)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.act3(self.layer3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output(x))
        return x
    

# Clase con red neuronal convolucional
class ConvNet(nn.Module):
    def __init__(self, shape):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=shape, out_channels=64, kernel_size=1)  
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=1)  
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=65, kernel_size=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=1)
        self.fc1 = nn.Linear(65, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2048, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 65)
        self.relu_fc3 = nn.ReLU()
        self.fc4 = nn.Linear(65, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        x = self.relu_fc3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


# Clase con red neuronal recurrente
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        out = self.sigmoid(out) 
        return out
model = RNN(150, 128, 2, 1)

def train_dnn_model(train_loader, X_test, y_test, lr, trainLoad =1):
    model = Deep(X_test.shape[1])
    loss_list = []
    if trainLoad == 1:
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Entrenar el modelo
        num_epochs = 10000
        for epoch in range(num_epochs):
            for inputs, labels in train_loader: 
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = 0
            # Imprimir la pérdida cada 10 epochs
            if (epoch + 1) % 1 == 0:

                loss_list.append(loss.item())
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')


            # Save the model weights
        torch.save(model.state_dict(), 'models/dnn_model.pth')
    else:
        model.load_state_dict(torch.load('models/dnn_model.pth'))
        model.eval()  # Poner el modelo en modo de evaluación
    # Test the model
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        print(f'Accuracy: {accuracy:.4f}')
    return model ,loss_list


def train_convnet_model(train_loader, X_test, y_test, lr, trainLoad =1):
    print(X_test.shape[1])
    model = ConvNet(X_test.shape[1])
    loss_list = []
    if trainLoad == 1:
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Train the model
        num_epochs = 100
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                if (inputs.size(0)==65):
                    # Forward pass
                    outputs = model(inputs.unsqueeze(2))
                    loss = criterion(outputs.squeeze(), labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Print the loss every 10 epochs
            if (epoch + 1) % 1 == 0:
                loss_list.append(loss.item())
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the model weights
        torch.save(model.state_dict(), 'models/cnn_model.pth')
    else:
        model.load_state_dict(torch.load('models/convnet_model.pth'))
        model.eval()
    # Test the model
    if (2==1):
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Accuracy: {accuracy:.4f}')
    return model ,loss_list


def train_rnn_model(train_loader, X_test, y_test, lr, trainLoad =1):
    
    
    loss_list = []
    if trainLoad == 1:
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)

        # Train the model
        num_epochs = 10000
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                # Forward pass
                outputs = model(inputs).squeeze()
                labels = labels.float()
                loss = criterion(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print the loss every 10 epochs
            if (epoch + 1) % 1 == 0:
                loss_list.append(loss.item())
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the model weights
        torch.save(model.state_dict(), 'models/rnn_model.pth')
    else:
        model.load_state_dict(torch.load('models/rnn_model.pth'))
        model.eval()
    # Test the model
    if (2==1):
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f'Accuracy: {accuracy:.4f}')
    return model ,loss_list



# Plot the learning curve
'''
# Plot the learning curve
plt.plot(num_epochs, loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()
'''

'''
list_of_files = [
        'BTCUSDCHistData.parquet',
        'ETHUSDCHistData.parquet',]

train_loader, X_test, y_test = load_data(list_of_files, 65)
#dnn_model, dnn_loss = train_dnn_model(train_loader, X_test, y_test, 0.00001, 1)
#rnn_model, rnn_loss = train_rnn_model(train_loader, X_test, y_test, 0.00001, 1)
cnn_model, cnn_loss = train_convnet_model(train_loader, X_test, y_test, 0.00011, 1)

from torchsummary import summary
#print(dnn_model)
#summary(dnn_model,  input_size=(1,X_test.shape[1]))
print(rnn_model)
summary(rnn_model, input_size=(65, 65, 150))
'''


