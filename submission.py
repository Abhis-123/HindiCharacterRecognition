import json
from predict import predict
from networks import CharacterRecognizer
import torch
from common import load_prediction_dataframe,transform


# from utils.io import write_json

def write_json(filename, result):

    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename):
    res = {}
    for i in range(1,100):
        test_set = str(i) + '.png'
        res[test_set] = 3
    write_json(filename, res)
    

if __name__ == '__main__':
    test_images_folder = "./dataset/test"
    test_df = load_prediction_dataframe(test_images_folder)
    net = CharacterRecognizer()
    path = "./weights/net.pt"
    net.load_state_dict(torch.load(path))
    df = predict(test_df, net,batch_size=1,transform=transform)
    d = {}
    for i,row in df.iterrows():
        d[row['image_name']]=row['target']
    filename = "./submission/submission.json"
    write_json(filename,d)

