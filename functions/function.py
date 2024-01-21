
import csv

def read_csv(csv_file, has_labels=True):
    image_paths = []
    labels = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        fundus_index = header.index("image_fundus")

        for row in reader:
            image_paths.append(row[fundus_index])

            if has_labels:
                label_index = header.index("label")
                labels.append(row[label_index])

    return image_paths, labels if has_labels else image_paths
    
    
if __name__ == "__main__":
    img_path= read_csv("/home/manish/Desktop/Project Optic/Project_opthalmology/Data/processed_train_ODIR-5K.csv")
    print(img_path)
    
    # with open(csv_file, 'r') as fil:
    #     reader = csv.reader(fil)
    #     next(reader)
    #     for row in reader:
    #         image_path.append(row[1])
    #         label.append(row[2])
            