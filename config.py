import torch

APP_NAME = "All Rounder AI API"

APP_VERSION = "4.0.0"

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CACHE_DIR = "/data/hf-cache"

MAX_INPUT_TOKENS = 4096

MAX_OUTPUT_TOKENS = 256