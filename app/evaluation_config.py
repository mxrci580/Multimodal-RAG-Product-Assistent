"""
Central Evaluation & Experiment Configuration File
Single source of truth for evaluation parameters, model identifiers,
data paths, and hyper-parameters across all retrieval and generation phases.
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Data Files
CATALOG_CSV_PATH = DATA_DIR / "amazon.csv"
IMAGE_CACHE_DIR = DATA_DIR / "image_cache"
MODELS_CACHE_DIR = DATA_DIR / "models_cache"

# Benchmark Datasets
GOLDEN_DATASET_V1_PATH = EVALUATION_DIR / "golden_dataset" / "questions.json"
GOLDEN_DATASET_V2_50_PATH = EVALUATION_DIR / "golden_dataset" / "questions_v2_50.json"
GOLDEN_DATASET_V2_50_CSV = EVALUATION_DIR / "golden_dataset" / "questions_v2_50.csv"

# Result Files
PHASE4B_RETRIEVAL_RESULTS_PATH = EVALUATION_DIR / "results" / "phase4b_retrieval_benchmark_results.json"
PHASE4C_GENERATION_RESULTS_PATH = EVALUATION_DIR / "results" / "phase4c_generation_evaluation_results.json"
PHASE4D_STATISTICAL_RESULTS_PATH = EVALUATION_DIR / "results" / "phase4d_statistical_validation_results.json"

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemini-flash-lite-latest")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

# ChromaDB
VECTOR_DB_PATH = str(PROJECT_ROOT / "vector_db")
COLLECTION_NAME = "products"

# Retrieval Hyperparameters
STAGE1_CANDIDATE_POOL_K = 20
RRF_K_CONSTANT = 60
RETRIEVAL_TOP_K = 5
GENERATION_CONTEXT_TOP_K = 3

# Statistical Hyperparameters
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 42

# Relevance Grades
RELEVANCE_SCALE = {
    3: "Ideal / Target Match",
    2: "Relevant / High-Quality Alternative",
    1: "Partially Relevant / Marginal",
    0: "Irrelevant / Trap"
}
