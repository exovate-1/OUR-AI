import requests
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
# Model Hyperparameters (Ultra-lightweight)
BLOCK_SIZE = 128
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
FF_DIM = N_EMBD * 4

# Data Streaming URL (SQuAD for Instruction Fine-Tuning)
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
MAX_TRAINING_ENTRIES = 5000

# Web Search Configuration (Used for RAG at the end)
GOOGLE_SEARCH_URL = "https://www.google.com/search?q=" 
# NOTE: Real-world RAG uses dedicated APIs (like SerpAPI or Tavily) for better results. 
# This simple scrape is for demonstration and confirmation only.

# --- GLOBAL VOCABULARY SETUP ---
qa_dataset, char_to_int, VOCAB_SIZE, DATA_SIZE = None, None, 0, 0
int_to_char = {} # Will be built after char_to_int is finalized

def get_qa_pairs_and_vocab(url, max_entries):
    """Streams data, extracts Q&A, and builds vocabulary."""
    global char_to_int, VOCAB_SIZE, int_to_char
    print(f"-> Streaming JSON data from URL: {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        data = response.json()
    except Exception as e:
        print(f"Error fetching/decoding data: {e}"); return None
        
    qa_list = []
    full_text = ""
    
    for document in data.get('data', []):
        for paragraph in document.get('paragraphs', []):
            for qa_pair in paragraph.get('qas', []):
                question = qa_pair.get('question', '')
                answer_text = qa_pair.get('answers', [{}])[0].get('text', '')
                if question and answer_text:
                    qa_list.append((question, answer_text))
                    # Add Q, separator, A, and newline to the vocabulary text
                    full_text += question + "|" + answer_text + "\n" 
                    if len(qa_list) >= max_entries: break
            if len(qa_list) >= max_entries: break
        if len(qa_list) >= max_entries: break

    # Build Character-Level Vocabulary and Special Tokens
    chars = sorted(list(set(full_text)))
    char_to_int = {ch: i + 3 for i, ch in enumerate(chars)} 
    char_to_int['<PAD>'] = 0
    char_to_int['|'] = 1  # Separator
    char_to_int['\n'] = 2 # End-of-Sequence

    VOCAB_SIZE = len(char_to_int)
    int_to_char = {i: ch for ch, i in char_to_int.items()}
    
    print(f"-> Extracted {len(qa_list)} Q&A pairs. Vocab Size: {VOCAB_SIZE}")
    return qa_list

# Execute Data Loading
qa_dataset = get_qa_pairs_and_vocab(SQUAD_URL, MAX_TRAINING_ENTRIES)
if not qa_dataset: exit()
