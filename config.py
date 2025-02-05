"""
config.py

Configuration settings for the MoE project.
"""

import os
import random
import numpy as np
import torch
import openai
from typing import Dict, Any

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DB_FILE = "sp500_daily_news.db"  # Path to your SQLite DB

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# WRDS credentials
WRDS_USERNAME = os.getenv("WRDS_USERNAME")
WRDS_PASSWORD = os.getenv("WRDS_PASSWORD")

# Data paths
DATA_DIR = "data"
CACHE_DIR = ".cache"
LOG_DIR = "logs"
RESULTS_DIR = "results"

# Ensure required directories exist
for dir_path in [DATA_DIR, CACHE_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "hidden_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "batch_size": 32,
}

# GPT configuration
GPT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 150,
}
