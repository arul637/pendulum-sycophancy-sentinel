# config.py

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"

# Generation parameters
TEMPERATURE = 0.1          # Low temperature for deterministic, factual answers
TOP_P = 0.9
NUM_CTX = 8192

# Anti-sycophancy settings
# CAAP: Compression Aware Anchor Preservation
# ISTM: Interference Time Sycophancy Mitigation
CAAP_ENABLED = True
ISTM_ENABLED = True

# Anchor extraction: ask model to describe objective visual features BEFORE answering
CAAP_ANCHOR_PROMPT = (
    "Describe ONLY the objective visual features of this image. "
    "List: main subject(s), colors, shapes, objects present, scene context. "
    "Do NOT guess, do NOT name the subject if uncertain. "
    "Be literal and factual. Output a short bullet list."
)

# ISTM: detection keywords that signal user pressure / leading questions
ISTM_PRESSURE_MARKERS = [
    "i think this is",
    "isn't this",
    "don't you think",
    "i'm sure this is",
    "this is actually",
    "it looks like a",
    "that's a",
    "that is a",
    "surely this is",
    "you're wrong",
    "no it's",
    "actually it's",
    "prove me wrong",
    "i believe this is",
    "i'm pretty sure",
    "obviously this is",
    "clearly this is",
]

# System prompt enforcing non-sycophantic behavior
SYSTEM_PROMPT = (
    "You are a factual vision assistant. Your sole duty is to report what is "
    "actually visible in the image, based on objective visual evidence.\n\n"
    "RULES:\n"
    "1. NEVER agree with the user if their claim contradicts visual evidence.\n"
    "2. If the user suggests a wrong label, explicitly correct them.\n"
    "3. Base every answer on the objective visual anchor description provided.\n"
    "4. If uncertain, say 'I cannot determine this with certainty from the image.'\n"
    "5. Do not be polite at the expense of truth. Accuracy > agreeableness."
)