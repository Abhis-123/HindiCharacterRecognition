import torch.optim as optim
from torch import nn
import torch
from networks import CharacterRecognizer
from dataset import CharacterDataset
from common import load_train_dataframe,transform
from torch.utils.data import DataLoader


def trainer(training_data,net,optimizer,loss,num_epochs):
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = CharacterRecognizer()
    #net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_df = load_train_dataframe("./dataset/training")
    dataset = CharacterDataset(train_df,transform=transform)
    l = dataset.__len__() 
    #train_set, test_set = torch.utils.data.random_split(dataset, [int(l*0.7), int(l*0.3)])
    train_loader = DataLoader(dataset,batch_size=32,shuffle=True)
    trainer(train_loader,net,optimizer,criterion,2)

