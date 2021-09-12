import torch
from dataset import PredictionDataset
from common  import load_prediction_dataframe,transform
from torch.utils.data import DataLoader
from utils.progressbar import Progressbar
import pandas as pd
from networks import CharacterRecognizer

# look for gpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediction_data_loader(prediction_dataframe,batch_size=1,transform=None):
    dataset = PredictionDataset(prediction_dataframe,transform=transform)
    loader = DataLoader(dataset,batch_size)
    return loader


def predict(prediction_dataframe,net,batch_size=1,transform=None,threshold=0.5):
    loader = prediction_data_loader(prediction_dataframe,batch_size=batch_size,transform=transform)
    num_steps = len(loader) 
    pgbar=Progressbar(num_steps)
    net.to(device)
    predictions =[]
    current_step = 0
    for i, data in enumerate(loader):
        inputs, paths= data
        inputs= inputs.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            outputs = (outputs>=threshold).float()
        for j in range(len(outputs)):
            predictions.append({
                'image_name':paths[j].split('/')[-1],
                'target':int(outputs[j].item())
            })
        current_step= current_step+1
        pgbar.update(current_step)

    return pd.DataFrame(predictions)




if __name__ == '__main__':

    test_images_folder = "./dataset/test/5.jpg"
    test_df = load_prediction_dataframe(test_images_folder)
    net = CharacterRecognizer()
    path = "./weights/net.pt"
    net.load_state_dict(torch.load(path))
    df = predict(test_df, net,batch_size=1,transform=transform)
    print(df.head())

