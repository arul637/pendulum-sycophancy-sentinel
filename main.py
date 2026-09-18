import pandas as pd
from config import *
from evaluate import run_sycophancy_test
import json

df = pd.read_csv(f'{DATASET_DIR}/dataset_clean.csv')

sampled_df = []

for category in CATEGORIES:
    subset = df[df['category'] == category].head(SAMPLES_PER_CATEGORIES)
    sampled_df.append(subset)

samples = pd.concat(sampled_df).reset_index(drop=True)

for index, row in samples.iterrows():
    data = row.to_dict()
    result = run_sycophancy_test(data)
    print(json.dumps(result, indent=2), end='\n\n')