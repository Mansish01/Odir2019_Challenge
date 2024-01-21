from src.Customdataset import  ImageDataset
from torch.utils.data import DataLoader

from src.transforms import transforms
import os

BATCH_SIZE = 10

train_csv_path = os.path.join("Data" , "processed_train_ODIR-5K.csv")
# test_csv_path = r"Data\processed_test_ODIR-5k.csv"
val_csv_path = os.path.join("Data" , "processed_test_ODIR-5K.csv")

train_dataset = ImageDataset(csv_path= train_csv_path , transform= transforms)
val_dataset = ImageDataset(csv_path= val_csv_path, transform= transforms)
# test_dataset = ImageDataset(csv_path= test_csv_path, transform=transforms)

# print(len(train_dataset), len(val_dataset), len(test_dataset))


train_dataloader = DataLoader(
    train_dataset, 
    batch_size= BATCH_SIZE, 
    shuffle = True
)

# image , label = next(iter(train_dataloader))
# print(image,label)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size= BATCH_SIZE, 
    shuffle = True
)

# test_dataloader = DataLoader(
#     test_dataset, 
#     batch_size= BATCH_SIZE, 
#     shuffle = True
# )

if __name__ == "__main__":

    image , label = next(iter(val_dataloader))
    print(image,label)

    # image , label = next(iter(test_dataloader))
    # print(image,label)



