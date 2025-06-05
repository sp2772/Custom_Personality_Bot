# import pandas as pd
# from tsundere import chat_data  # Replace 'name' with the actual filename (without .py)
# import csv

# def save_to_csv(output_file="conversation_dataset_ShirayukiV2.csv"):
#     L=[]
#     for tup in chat_data:
#         L.append({'guy':tup[0],'girl':tup[1]})
        
#     with open(output_file,'w', newline='',encoding='utf-8') as f:
#         w=csv.DictWriter(f, fieldnames=['guy','girl'])
#         w.writeheader()
#         w.writerows(L)

# if __name__ == "__main__":
#     save_to_csv()

import pandas as pd
from new_tuples import chat_data  
import csv

def save_to_csv(output_file="new_tuples.csv"):
    L = [{'guy': tup[0], 'girl': tup[1]} for tup in chat_data]

    # Save the initial dataset
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['guy', 'girl'])
        w.writeheader()
        w.writerows(L)

    # Shuffle the dataset
    df = pd.read_csv(output_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle rows
    df.to_csv(output_file, index=False)  # Save shuffled data

if __name__ == "__main__":
    save_to_csv()
