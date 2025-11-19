import requests
import json
import numpy as np
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
BLOCK_SIZE = 128
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
HEAD_SIZE = N_EMBD // N_HEAD
VOCAB_SIZE = 0
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
MAX_TRAINING_ENTRIES = 5000
GOOGLE_SEARCH_URL = "https://www.google.com/search?q=" 
# NOTE: Parameters (W, B) will be initialized randomly, simulating a trained model.

# --- UTILITIES ---

def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def matmul_with_mask(Q, K, V, mask):
    """Performs Scaled Dot-Product Attention (Forward Pass)."""
    # 1. Attention Scores: Q @ K^T / sqrt(d_k)
    d_k = K.shape[-1]
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k) # (B, H, T, T)

    # 2. Causal Masking (prevents looking into the future)
    # The mask is upper triangular (-inf)
    if mask is not None:
        scores = scores + mask 

    # 3. Softmax
    weights = softmax(scores) # (B, H, T, T)
    
    # 4. Aggregate Values
    output = weights @ V # (B, H, T, T) @ (B, H, T, C) -> (B, H, T, C)
    return output

# --- WEIGHT & PARAMETER CLASS (No backprop implemented) ---
class Weights:
    """A minimal class to hold 'trained' NumPy parameters."""
    def __init__(self, shape):
        # Glorot/Xavier initialization: variance scaled by input/output fan
        self.W = np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
        self.B = np.zeros((shape[-1],)) # Bias initialized to zero
    
    def __call__(self, x):
        return x @ self.W + self.B

class LayerNorm:
    """Simple Layer Normalization implementation."""
    def __init__(self, size):
        self.gamma = np.ones((size,))
        self.beta = np.zeros((size,))
        self.eps = 1e-5

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# --- MODEL ARCHITECTURE (The GPT Model) ---
class SelfAttentionHead:
    """A single attention head using only NumPy operations."""
    def __init__(self, n_embd, head_size):
        self.key = Weights((n_embd, head_size))
        self.query = Weights((n_embd, head_size))
        self.value = Weights((n_embd, head_size))
        # Mask is created once (T=BLOCK_SIZE, T=BLOCK_SIZE)
        self.mask = np.triu(np.ones((BLOCK_SIZE, BLOCK_SIZE)) * -np.inf, k=1)
    
    def __call__(self, x):
        T, C = x.shape[1:]
        
        # Project Q, K, V
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # Compute Attention (only for the single head)
        output = matmul_with_mask(q, k, v, self.mask[:T, :T])
        
        return output

class MultiHeadAttention:
    """Combines multiple heads and projects the output."""
    def __init__(self, n_embd, n_head):
        self.heads = [SelfAttentionHead(n_embd, HEAD_SIZE) for _ in range(n_head)]
        self.proj = Weights((n_embd, n_embd))

    def __call__(self, x):
        # Stack the outputs from all heads
        head_outputs = np.stack([h(x) for h in self.heads], axis=1) # (B, H, T, C)
        
        # Reshape to (B, T, N_EMBD) for final projection
        B, H, T, C = head_outputs.shape
        x_stacked = head_outputs.transpose(0, 2, 1, 3).reshape(B, T, N_EMBD)
        
        return self.proj(x_stacked)

class FeedFoward:
    """Two-layer MLP."""
    def __init__(self, n_embd):
        FF_DIM = n_embd * 4
        self.net1 = Weights((n_embd, FF_DIM))
        self.net2 = Weights((FF_DIM, n_embd))
        
    def __call__(self, x):
        # ReLU activation (np.maximum(0, x))
        return self.net2(np.maximum(0, self.net1(x)))

class Block:
    """A single Transformer Block."""
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def __call__(self, x):
        # Pre-Norm + Residual Connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleNumPyGPT:
    """The full, tiny LLM in NumPy."""
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head):
        # Learnable Embeddings
        self.token_embedding_table = Weights((vocab_size, n_embd))
        self.position_embedding_table = Weights((block_size, n_embd))
        
        # Stack of Transformer Blocks
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        
        # Final Norm and Head
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Weights((n_embd, vocab_size))

    def __call__(self, idx):
        # idx is (B, T)
        T = idx.shape[1]
        
        # Embeddings
        tok_emb = self.token_embedding_table.W[idx] 
        pos_emb = self.position_embedding_table.W[np.arange(T)]
        x = tok_emb + pos_emb # (B, T, C)
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Final Norm and Linear Head
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)
        return logits
