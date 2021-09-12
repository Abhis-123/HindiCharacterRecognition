from torch.utils.data import Dataset
from torchvision.io import read_image
from common import load_train_data

class CharacterDataset(Dataset):
    def __init__(self,dataframe=None,transform=None):
        """
        parameters:
            dataframe : a pandas dataframe for reading files it must have 'path' and 'label' attributes
            transform : function to transform the image data 

        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image= read_image(row['path'])
        label= row['label']

        if self.transform:
            image= self.transform(image)
        return image,label   

if __name__ == '__main__':
    df = load_train_data("dataset/training")
    dataset = CharacterDataset(df)
    print(dataset.__getitem__(1)[0])

