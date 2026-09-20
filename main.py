import pandas as pd
from config import *
from evaluate import run_batch

df = pd.read_csv(f'{DATASET_DIR}/dataset_clean.csv')

sampled_df = []

for category in CATEGORIES:
    subset = df[df['category'] == category].head(SAMPLES_PER_CATEGORIES)
    sampled_df.append(subset)

samples = pd.concat(sampled_df).reset_index(drop=True)

dataset = samples.to_dict(orient="records")

run_batch(dataset)