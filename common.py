import pandas as pd
import os

def load_train_data(directory):
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

def prediction_data(path):
    image_label_list = [] 
    if os.path.isdir(path):
        images = os.listdir(f'{path}')
        for image in images:
            if os.path.isfile(f"{path}/{image}"):
                image_label_list.append({'path':f"{path}/{image}" , 'label':None})
    if os.path.isfile(path):
        image_label_list.append({'path':f"{path}" , 'label':None})
    return pd.DataFrame(image_label_list)









if __name__ == "__main__":
    print(load_train_data("./dataset/training").head(4))