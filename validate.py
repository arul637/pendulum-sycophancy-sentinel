import pandas as pd 
from config import * 
from langchain_core.prompts import PromptTemplate


df = pd.read_csv(f'{DATASET_DIR}/dataset_clean.csv')

sampled_df = []


for category in CATEGORIES:
    subset = df[df['category'] == category].head(SAMPLES_PER_CATEGORIES) 
    sampled_df.append(subset)

samples = pd.concat(sampled_df)

# prompt_template = ''
# with open('prompts/vision_prompt.txt', 'r') as file:
#     prompt_template = file.read()

# prompt = PromptTemplate(
#     template=prompt_template,
#     input_variables=['question', 'output_format']
# )

# formatted_prompt = prompt.format(
#     question='',
#     output_format=''
# )




