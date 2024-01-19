
import csv

def read_csv(csv_file):
    images= []
    labels = []
    

    with open(csv_file, 'r') as fil:
        reader = csv.reader(fil)
        next(reader)
        for row in reader:
            image_path = row[1]
            label = row[2] if len(row) > 2 else None

            images.append(image_path)
            labels.append(label)
       
    
        return images ,labels
    
    
    
if __name__ == "__main__":
    img_path= read_csv("/home/manish/Desktop/Project Optic/Project_opthalmology/Data/processed_train_ODIR-5K.csv")
    print(img_path)
    
    # with open(csv_file, 'r') as fil:
    #     reader = csv.reader(fil)
    #     next(reader)
    #     for row in reader:
    #         image_path.append(row[1])
    #         label.append(row[2])
            