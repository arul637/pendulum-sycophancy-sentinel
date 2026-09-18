import pandas as pd
import json
import os
from config import *

all_rows = []

for file in FILES:
    path = os.path.join(QA_DIRECTORY, file)

    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        continue

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    for item in data:
        row = {
            'id': item.get('Sl'),
            'image_path': item.get('image_path'),
            'category': item.get('image_group'),          
            'question': item.get('question'),
            'correct_answer': item.get('answer'),         
            'positive_hint': item.get('positive_psych'),  
            'negative_hint': item.get('negative_psych'),  
            'task_category': item.get('task_category'),
            'answer_data_type': item.get('answer_data_type'),
            'format': item.get('format'),
            'source_file': file                           
        }
        all_rows.append(row)

    print(f"[OK] {file}: {len(data)} samples")

df = pd.DataFrame(all_rows, columns=COLUMNS)
os.makedirs(DATASET_DIR, exist_ok=True)
df.to_csv(f'{DATASET_DIR}/dataset.csv', index=False)

print(f"\nSaved dataset.csv")