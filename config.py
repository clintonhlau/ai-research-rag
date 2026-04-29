import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
LLM_MODE = "claude"                    # "claude" or "local"
CLAUDE_MODEL = "claude-sonnet-4-5"
LOCAL_MODEL = "llama3.1:8b"
MAX_TOKENS = 1024
TEMPERATURE = 0.1

# --- Embedding (locked after first index build) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Reranking ---
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Chunking ---
CHUNK_SIZE = 500                       # tiktoken tokens
CHUNK_OVERLAP = 50                     # 10% context overlap between chunks

# --- Retrieval ---
TOP_K_RETRIEVAL = 10                   # pre-reranking
TOP_K_RERANKED = 5                     # post-reranking

# --- ArXiv ---
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.MA", "cs.CY"] # AI, ML, Computation & Language, Multiagent Systems, Computers & Society
ARXIV_SAFETY_KEYWORDS = [
    "alignment", "value alignment", "AI safety", "RLHF",
    "reinforcement learning from human feedback", "constitutional AI",
    "interpretability", "mechanistic interpretability", "explainability",
    "red teaming", "jailbreak", "prompt injection", "adversarial",
    "robustness", "reward hacking", "reward modeling",
    "hallucination", "factuality", "harmlessness", "refusal",
    "existential risk", "AI governance", "AI risk"
]
ARXIV_MONTHS_BACK = 12
ARXIV_RATE_LIMIT_SLEEP = 5            # seconds between ArXiv API page requests
ARXIV_NUM_RETRIES = 5                 # retries per page on 429
ARXIV_BACKOFF_BASE = 2                # exponential backoff multiplier on 429
SECTIONS_TO_EXTRACT = ["abstract", "introduction", "results", "conclusion"]

# --- GROBID ---
GROBID_URL = "http://localhost:8070"
GROBID_TIMEOUT = 60                   # seconds

# --- Storage ---
DB_PATH = "data/papers.db"
CHROMA_PATH = "data/chroma_db"
CHROMA_COLLECTION = "arxiv_papers"

# --- API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL")