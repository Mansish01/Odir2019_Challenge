import pandas as pd
import os

BASE_DIR= r"Data"
file_path = r"Data/ODIR-5K_Training_Annotations(Updated)_V2.xlsx"
annotated_data = pd.read_excel(file_path)
csv_file= annotated_data.to_csv(os.path.join(BASE_DIR, "Data.csv"), index= False)

