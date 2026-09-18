# Vision Sycophancy Evaluation

Evaluating vision-language models for sycophantic behavior on the Pendulum dataset using Ollama and LangChain.

## Setup

install the python dependencies using requirement.txt

```
pip install -r requirements.txt
```

download ollama and pull any vision model which suit for your operating system 
```
ollama pull gemma3:4b
```


## Pipeline

Follow these steps in order:

### 1. Convert JSON to CSV

```
    python3 convert_csv.py
```

Converts the raw JSON dataset files into a single CSV stored in the `dataset/` directory.

### 2. Clean the Data

Open and run `cleaning.ipynb`.

This notebook handles duplicate removal and data cleaning on the generated CSV.

### 3. Configure Settings

Edit `config.py` to set your paths and experiment parameters:

- `DATASET_DIR` — path to the dataset directory
- `CATEGORIES` — list of task categories to evaluate
- `SAMPLES_PER_CATEGORIES` — number of samples per category
- `MODEL_NAME` — Ollama model to use

### 4. Run the Evaluation

```
    python3 main.py
```

Runs the sycophancy test on the sampled data and prints results.
