import torch
from torch import nn 



def conv_layer(in_channels, out_channels,kernel_size=3,padding=1,batchnorm=False,max_pool=False):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels,kernel_size,padding=padding))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    if max_pool:
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    layer =nn.Sequential(*layers)
    return layer


def Dense(in_channels, out_channels,activation=None):
    layers = []
    layers.append(nn.Linear(in_channels, out_channels))
    if activation is not None:
        layers.append(activation)
    layer= nn.Sequential(*layers)
    return layer
class CharacterRecognizer(nn.Module):
    def __init__(self,*args,**kwargs):
        super(CharacterRecognizer, self).__init__(*args,**kwargs)
        # convolution layers
        self.layer1= conv_layer(3,32,kernel_size=3,padding=1,batchnorm=True,max_pool=True)
        self.layer2= conv_layer(32,64,kernel_size=3,padding=1,batchnorm=False,max_pool=True)
        self.layer3= conv_layer(64,128,kernel_size=3,padding=1,batchnorm=True,max_pool=True)
        self.layer4= conv_layer(128,256,kernel_size=3,padding=1,batchnorm=False,max_pool=True)
        self.layer5= conv_layer(256,512,kernel_size=3,padding=1,batchnorm=True,max_pool=True)
        self.flatten = nn.Flatten()
        self.feature_layer =    Dense(512*64//2*2*2*2*2,1024,nn.Sigmoid())
        self.prediction_layer= Dense(1024,1,activation=nn.Sigmoid())

        


    def forward(self,input):
        x = self.layer1(input)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.layer4(x)
        x= self.layer5(x)
        x= self.flatten(x)
        x=self.feature_layer(x)
        x= self.prediction_layer(x)
        return x



if __name__ == "__main__":
    net = CharacterRecognizer()
    print(net)
