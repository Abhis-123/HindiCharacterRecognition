import pandas as pd
import os
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torch
def load_train_dataframe(directory):
    class_folders = os.listdir(directory)
    image_label_list = [] 
    for i in range(0, len(class_folders)):
        if os.path.isdir(f'{directory}/{class_folders[i]}'):
            images = os.listdir(f'{directory}/{class_folders[i]}')
            for image in images:
                if os.path.isfile(f"{directory}/{class_folders[i]}/{image}"):
                    image_label_list.append({'path':f"{directory}/{class_folders[i]}/{image}" , 'label':i})
    df = pd.DataFrame(image_label_list)
    return df

def prediction_dataframe(path):
    image_label_list = [] 
    if os.path.isdir(path):
        images = os.listdir(f'{path}')
        for image in images:
            if os.path.isfile(f"{path}/{image}"):
                image_label_list.append({'path':f"{path}/{image}" , 'label':None})
    if os.path.isfile(path):
        image_label_list.append({'path':f"{path}" , 'label':None})
    return pd.DataFrame(image_label_list)


def transform(tensor):
    if not torch.is_tensor(tensor):
        to_tensor = transforms([transforms.ToTensor()])
        tensor = to_tensor(tensor)

    if not torch.is_floating_point(tensor):
        tensor.type(torch.float32)

    transformations = []
    

    transformations.append(transforms.Normalize(0,1))
    
    t= transforms.Compose(transformations)
    return t(tensor)


if __name__ == "__main__":
    print(load_train_dataframe("./dataset/training").head(4))