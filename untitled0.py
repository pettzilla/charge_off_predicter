
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.nn as nn

import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader, random_split


class LoanData(Dataset):
    def __init__(self):
        df = pd.read_csv('credit_train.csv', delimiter=',', encoding="utf-8-sig")
        df = df.drop(['Loan ID','Customer ID'], axis=1)
        
        self.catagorical = ['Term','Years in current job','Home Ownership','Purpose','Months since last delinquent','Loan Status']
        df_dummies = (pd.get_dummies(data= df ,columns =self.catagorical ))
        #removing nan inputs
        df_dummies = df_dummies.apply(pd.to_numeric,errors='coerce')
        df_dummies = df_dummies.dropna()
        df_dummies =df_dummies.reset_index(drop = True)
        self.df_with_dummies = df_dummies

        self.target = 'Loan Status_Fully Paid'
        holdX = (self.df_with_dummies.drop(self.target, axis=1)).values
        # normalising x values
        min_max_scale = preprocessing.MinMaxScaler()
        X_scaled = min_max_scale.fit_transform(holdX)
        
        self.X = pd.DataFrame(X_scaled)
        self.Y= self.df_with_dummies[self.target]
    def __len__(self):                                
        return len(self.df_with_dummies)                    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):

            idx = idx.tolist()



        return [self.X.iloc[idx].values, self.Y[idx]]

class Net(nn.Module):
    def __init__(self):
         super().__init__()
         self.fc1 = nn.Linear(161, 225)
         self.fc2 = nn.Linear(225,100)
         self.fc3 = nn.Linear(100,30)
         self.fc4 = nn.Linear(30,30)
         self.fc5 = nn.Linear(30,10)
         self.fc6 = nn.Linear(10, 1)
         # drops reduce the probibility of overfitting
         self.drop1 = nn.Dropout(p=0.1)
         self.drop2 = nn.Dropout(p=0.2)



    def forward(self, x):

        x = self.fc1(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.drop1(x)
        x = self.fc3(x)
        x = self.drop2(x)
        x = self.fc4(x)

        x = self.fc5(x)
        x = self.drop1(x)
        x = self.fc6(x)

        #convert output between 0 and 1
        x= torch.sigmoid(x)
        return x
    

class train():
    dataset = LoanData()
    train_size = int(0.8 * len(dataset))
    
    test_size = len(dataset) - train_size

    trainset, testset = random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=200, shuffle=True)

    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    net = Net()
    net.to(device)
    Loss_function = nn.BCELoss()
    
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    loss_per_iter = []

    loss_per_batch = []
    #increesing epochs will lower error in both train and test 
    for epoch in range(40):



        running_loss = 0.0

        for i, (inputs, labels) in enumerate(trainloader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            p=inputs.float()


            # Zero the parameter gradients

            optimizer.zero_grad()



            # Forward + backward + optimize

            outputs = net.forward(inputs.float())

            loss = Loss_function(outputs, labels.unsqueeze(1).float())
           

            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            loss_per_iter.append(loss.item())


            # Save loss to plot

            
            


        loss_per_batch.append(running_loss / (i + 1))

        running_loss = 0.0



    # Comparing training to test

    dataiter = iter(testloader)

    inputs, labels = dataiter.next()

    inputs = inputs.to(device)

    labels = labels.to(device)

    outputs = net(inputs.float())
        
    ans = 0
    


    print("Training:", np.sqrt(loss_per_batch[-1]))
    loss = Loss_function(outputs, labels.unsqueeze(1).float())
           


    print("test :")
    print(loss.item())

    #compute acuracy
    pred = np.round(outputs.detach()).tolist()
    target = np.round(labels.detach()).tolist()
    print ("accuracy on test set is",accuracy_score(target, pred) )


    # Plot training loss curve

    plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per epoch")

    plt.plot(np.arange(len(loss_per_iter), step=323) + 3, loss_per_batch, ".-", label="Loss per mini-batch")

    plt.xlabel("Number of epochs")

    plt.ylabel("Loss")
    plt.legend()

    plt.show()
    
    

if __name__ == "__main__":
    train()
    

    