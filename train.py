import torch.optim as optim
from torch import nn
import torch
from networks import CharacterRecognizer
from dataset import CharacterDataset
from common import load_train_dataframe,transform
from torch.utils.data import DataLoader
from losses import Loss
from tqdm import tqdm
from metrics import accuracy
import time

# look for gpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trainer(training_data,validation_data,net,optimizer,loss_function,num_epochs):
    # moving everything to gpu if available
    net.to(device)


    for epoch in range(1,num_epochs+1):  # loop over the dataset multiple times
        t1 = time.time()
        training_loss = []
        train_predictions = []
        train_labels =[]
        print(f"epoch {epoch}/{num_epochs}  [=",end="")
        step = (len(training_data)+len(validation_data))//20
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
            train_predictions=train_predictions+output.tolist()
            train_labels=train_labels+ labels.tolist()
            training_loss.append(loss.item())
            if i%step==0:
                print("=" ,end="")

        validation_loss=[]
        validation_predictions = []
        validation_labels = []

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
            validation_predictions=validation_predictions+output.tolist()
            validation_labels=validation_labels+labels.tolist()
            validation_loss.append(loss.item())
            if i%step==0:
                print("=" ,end="")


        validation_acc="{:.3f}".format(accuracy(torch.Tensor(validation_predictions),torch.Tensor(validation_labels)))
        train_acc ="{:.3f}".format(accuracy(torch.Tensor(train_predictions),torch.Tensor(train_labels)))
        train_loss="{:.5f}".format(sum(training_loss)/len(training_loss))
        val_loss  ="{:.5f}".format(sum(validation_loss)/len(validation_loss))
        epoch_time="{:.3f}".format(time.time()-t1)
        print(f"=] epoch_time- {epoch_time} train_loss- {train_loss}s train_acc- {train_acc} val_loss- {val_loss} val_acc- {validation_acc} ")





if __name__ == '__main__':
    net = CharacterRecognizer()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_df = load_train_dataframe("./dataset/training")
    

    dataset = CharacterDataset(train_df,transform=transform)

    train_set, val_set = torch.utils.data.random_split(dataset, [5000,875])
    train_loader = DataLoader(train_set,batch_size=64,shuffle=True)
    val_loader = DataLoader(val_set,batch_size=64,shuffle=True)
    
    trainer(train_loader,val_loader,net,optimizer,Loss,20)

    import os
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    path = "./weights/net.pt"
    torch.save(net.state_dict(),path)

