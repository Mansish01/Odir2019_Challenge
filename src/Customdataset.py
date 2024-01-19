from torch.utils.data import Dataset
from os.path import join
import pandas as pd
import csv
from functions.function import read_csv
from PIL import Image

label_to_index_map = {'N':0,
   'D':1,
   'G':2,
    'C':3,
    'A':4,
    'H':5,
    'M':6,
    '0':7}

def label_to_index(label):
    return label_to_index_map.get(label, 7)

class ImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        
        images, labels = read_csv(csv_path)
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __str__(self):
        return f"the number of samples{self.__len__()}"


    def __getitem__(self, index):
        image_name = self.images[index]
        label_name = self.labels[index]
        image_path= join("Data", "ODIR-5K_Training_Dataset", image_name)
        
        image = Image.open(image_path).convert("RGB")
        label = label_to_index(label_name)
        
        if self.transform:
            image = self.transform(image)

        return image, label

        
        
        
if __name__ == "__main__":

    label=label_to_index('M')
    print(label)

    