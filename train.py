import torch.optim as optim
import numpy as np
import torch
from networks import CharacterRecognizer
from dataset import CharacterDataset
from common import load_train_dataframe,transform
from torch.utils.data import DataLoader
from losses import Loss
from metrics import accuracy
from utils.progressbar import Progressbar


# look for gpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trainer(training_data,validation_data,net,optimizer,loss_function,num_epochs):
    # moving everything to gpu if available
    net.to(device)
    g_training_loss = []
    g_training_accuracy = []
    g_validation_loss=[]
    g_validation_accuracy = []

    for epoch in range(1,num_epochs+1):  # loop over the dataset multiple times
        training_loss = []
        training_accuracy = []
        validation_loss=[]
        validation_accuracy = []

        print(f"epoch {epoch}/{num_epochs} ")
        total_batches = len(training_data)+len(validation_data)
        pbar = Progressbar(target=total_batches)

        current_batch =0
        for i, data in enumerate(training_data,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs,torch.reshape(labels,[labels.shape[0],1]))
            loss.backward()
            optimizer.step()
            # store statistics
            output = torch.reshape(outputs,[outputs.shape[0]])
            training_loss.append(loss.item())
            acc = accuracy(output,labels)
            training_accuracy.append(acc.item())
            current_batch=current_batch+1
            pbar.update(current_batch,values =[('train_loss',loss.item()),('train_accuracy',acc.item())])
            
        for i, data in enumerate(validation_data,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # move to gpu
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            with torch.no_grad():
                # forward + backward + optimize
                outputs = net(inputs)
                loss = loss_function(outputs,torch.reshape(labels,[labels.shape[0],1]))

            # print statistics
            output = torch.reshape(outputs,[outputs.shape[0]])

            validation_loss.append(loss.item())
            val_acc = accuracy(output,labels)
            validation_accuracy.append(val_acc.item())
            current_batch =current_batch + 1
            pbar.update(current_batch,values = [('val_loss',loss.item()),('val_acc',val_acc.item())])
        # print(training_accuracy[0],training_loss[0],validation_accuracy[0],validation_loss[0])

        g_training_loss.append(torch.mean(torch.tensor(g_training_loss)))
        g_training_accuracy.append(torch.mean(torch.tensor(training_accuracy)))
        g_validation_loss.append(torch.mean(torch.tensor(validation_loss)))
        g_validation_accuracy.append(torch.mean(torch.tensor(validation_accuracy)))

    return {
            'train_accuracy': g_training_accuracy,
            'train_loss': g_training_loss,
            'validation_accuracy': g_validation_accuracy,
            'validation_loss':g_training_loss}
        





if __name__ == '__main__':
    net = CharacterRecognizer()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_df = load_train_dataframe("./dataset/training")
    

    dataset = CharacterDataset(train_df,transform=transform)

    train_set, val_set = torch.utils.data.random_split(dataset, [5000,875])
    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=64,shuffle=True)
    
    trainer(train_loader,val_loader,net,optimizer,Loss,25)
    
    import os
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    path = "./weights/net.pt"
    save_weights=False
    if save_weights:
        torch.save(net.state_dict(),path)

