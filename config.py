QA_DIRECTORY = 'QA'

FILES = [
    'camouflaged.json',
    'confusing_perspective.json',
    'OCR.json',
    'puzzle.json',
    're_used.json',
    'self_captured.json'
]

COLUMNS = [
    'id',
    'image_path',
    'category',
    'question',
    'correct_answer',
    'positive_hint',
    'negative_hint',
    'task_category',
    'answer_data_type',
    'format',
    'source_file'
]

DATASET_DIR = 'dataset'
IMAGES_ROOT = 'images'
OUTPUT_CSV = 'sycophancy_results.csv'

CATEGORIES = [
    'OCR',
    'camouflaged',
    'confusing_perspective',
    'puzzle',
    're_used',
    'self_captured'
]

SAMPLES_PER_CATEGORIES = 1

OLLAMA_IMAGE_MODEL = 'gemma3:4b'
OLLAMA_URL = 'http://localhost:11434/api/generate'
TIMEOUT = 120