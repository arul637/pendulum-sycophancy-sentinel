# PENDULUM Dataset Preparation

Convert raw PENDULUM JSON files into a single CSV and clean it for the multi-turn sycophancy experiment.

## Files

| File | Purpose |
|------|---------|
| convert_csv.py | Reads all 6 JSON files from QA/ and merges them into dataset.csv |
| cleaning.ipynb | Cleans dataset.csv (NaN, duplicates, whitespace) and saves dataset_clean.csv |

## Step 1: Convert JSON to CSV

Run this command in your terminal:

    python convert_csv.py

Output: dataset.csv with these columns:
- id, image_path, category, question, correct_answer
- positive_hint, negative_hint, task_category
- answer_data_type, format, source_file

## Step 2: Clean the Dataset

Open cleaning.ipynb in Jupyter or Google Colab and run all cells.

What it does:
1. Loads dataset.csv
2. Checks missing values and duplicates
3. Drops rows with NaN in format and answer_data_type
4. Fills non-critical NaN with empty strings
5. Removes duplicates (full row + image_path)
6. Strips whitespace and normalizes case
7. Shows insights (category counts, answer types, plots)
8. Saves dataset_clean.csv

Output: dataset_clean.csv

## Requirements

Install the required packages:

    pip install pandas numpy matplotlib seaborn

## Dataset Columns

| Column | Description |
|--------|-------------|
| id | Original sample index (Sl) |
| image_path | Filename of the image |
| category | Image group (OCR, camouflaged, etc.) |
| question | Question asked about the image |
| correct_answer | Ground truth answer |
| positive_hint | Hint supporting the correct answer |
| negative_hint | Hint pushing toward the wrong answer |
| task_category | Task type (Adversarial, object, etc.) |
| answer_data_type | string / number / binary |
| format | Expected answer format |
| source_file | Original JSON filename |

## Notes

- negative_hint is used as the adversarial pressure in multi-turn conversations.
- Images must be unzipped separately and matched to image_path in the next phase.