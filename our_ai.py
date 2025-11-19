import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import json
import numpy as np

# --- CONFIGURATION (Hyperparameters) ---
# Model Configuration
BLOCK_SIZE = 128     # Maximum context length for the Transformer
N_EMBD = 128         # Embedding dimension (d_model)
N_HEAD = 4           # Number of attention heads
N_LAYER = 4          # Number of Transformer blocks
DROPOUT = 0.1        # Dropout rate

# Training Configuration
BATCH_SIZE = 32
MAX_ITERS = 10       # Use low iterations for quick test
LEARNING_RATE = 1e-4

# Data URL
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
MAX_TRAINING_ENTRIES = 5000

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# --- 1. DATA LOADING AND PREPARATION UTILITIES ---
# Global variables for vocabulary
char_to_int = {}
int_to_char = {}
VOCAB_SIZE = 0

def get_qa_pairs_and_vocab(url, max_entries):
    """Streams data, extracts Q&A, and builds a character-level vocabulary."""
    global char_to_int, int_to_char, VOCAB_SIZE
    print(f"-> Streaming JSON data from URL...")
    
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
                    full_text += question + "|" + answer_text + "\n" # Q | A \n
                    if len(qa_list) >= max_entries: break
            if len(qa_list) >= max_entries: break
        if len(qa_list) >= max_entries: break

    # Build Vocabulary and Special Tokens
    chars = sorted(list(set(full_text)))
    char_to_int = {ch: i + 3 for i, ch in enumerate(chars)} 
    char_to_int['<PAD>'] = 0
    char_to_int['|'] = 1  # Separator token
    char_to_int['\n'] = 2 # End-of-Sequence token

    VOCAB_SIZE = len(char_to_int)
    int_to_char = {i: ch for ch, i in char_to_int.items()}
    
    print(f"-> Extracted {len(qa_list)} Q&A pairs. Vocab Size: {VOCAB_SIZE}")
    return qa_list

def tokenize_qa(q, a, max_len):
    """Converts Q&A to a padded token sequence."""
    # Ensure separator and EOS tokens are in the dictionary
    tokens = [char_to_int.get(c, char_to_int['<PAD>']) for c in q] 
    tokens += [char_to_int['|']]
    tokens += [char_to_int.get(c, char_to_int['<PAD>']) for c in a]
    tokens += [char_to_int['\n']]
    
    # Truncate or Pad
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [char_to_int['<PAD>']] * (max_len - len(tokens))

    return torch.tensor(tokens, dtype=torch.long)

def get_batch(data_tuples, batch_size, block_size):
    """Creates a training batch (X=input, Y=target) from Q&A data."""
    # Select a random batch of indices
    ix = torch.randint(len(data_tuples), (batch_size,))
    
    X_batch = []
    Y_batch = []
    
    for i in ix:
        q, a = data_tuples[i]
        # We need a sequence of length (BLOCK_SIZE + 1) to create the input X and target Y
        full_tokens = tokenize_qa(q, a, block_size + 1)
        
        # X: Input sequence (tokens 0 to BLOCK_SIZE-1)
        # Y: Target sequence (tokens 1 to BLOCK_SIZE) - shifted by one
        X_batch.append(full_tokens[:-1])
        Y_batch.append(full_tokens[1:])
        
    X = torch.stack(X_batch).to(DEVICE)
    Y = torch.stack(Y_batch).to(DEVICE)
    return X, Y

# Execute Data Loading
qa_dataset = get_qa_pairs_and_vocab(SQUAD_URL, MAX_TRAINING_ENTRIES)
if not qa_dataset: exit()

# --- 2. PYTORCH GPT ARCHITECTURE ---

class Head(nn.Module):
    """One single head of the Causal Self-Attention mechanism."""
    def __init__(self, head_size):
        super().__init__()
        # Project input embedding into Query, Key, and Value vectors
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Causal mask: registered as a buffer so it's not a learned parameter
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (sequence length), Channel (head_size)
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        
        # Compute attention scores ("affinities")
        # wei = Q @ K^T / sqrt(d_k)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        
        # Causal Masking: For decoder-only, a token can only look at tokens before it.
        # Mask out upper triangle (future tokens) with -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        wei = F.softmax(wei, dim=-1) # (B, T, T) - Attention weights
        wei = self.dropout(wei)

        # Weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v    # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads running in parallel, then concatenated."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD) # Final projection layer
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Concatenate the outputs from all heads along the channel dimension (C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """A simple two-layer MLP for the transformer block."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Standard: 4x expansion factor
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer back to n_embd
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A single Transformer Block: Communication (Attention) followed by Computation (FFN)."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        # Sub-layer 1: Multi-Head Causal Self-Attention
        self.sa = MultiHeadAttention(n_head, head_size) 
        # Sub-layer 2: Feed Forward Network
        self.ffwd = FeedFoward(n_embd) 
        
        # LayerNorm applied *before* the attention and FFN (pre-norm style, common in modern GPTs)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Apply LayerNorm, then Attention, and add the residual connection (x + ...)
        x = x + self.sa(self.ln1(x))
        # Apply LayerNorm, then FFN, and add the residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleGPT(nn.Module):
    """The main Generative Pre-trained Transformer model."""
    def __init__(self):
        super().__init__()
        
        # 1. Token and Positional Embeddings
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # 2. Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        
        # 3. Final Layer Normalization and Head
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE) # Project to vocabulary size for logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are (B, T)
        
        # Embeddings: Tokens (B, T, C) + Positions (T, C)
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb # (B, T, C)
        
        # Transformer Blocks
        x = self.blocks(x)
        
        # Final Norm and Linear Head
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

        if targets is None:
            loss = None
        else:
            # Reshape for cross_entropy: (B*T, VOCAB_SIZE) and (B*T)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Initialize the Model
model = SimpleGPT()
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 3. TRAINING LOOP AND INFERENCE ---

print("\n--- Starting PyTorch Q&A Fine-Tuning ---")

for iter in range(MAX_ITERS):
    # Sample a batch of data
    xb, yb = get_batch(qa_dataset, BATCH_SIZE, BLOCK_SIZE)

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % (MAX_ITERS // 10) == 0 or iter == MAX_ITERS - 1:
        print(f"Step {iter}/{MAX_ITERS}: Train Loss = {loss.item():.4f}")

print("--- PyTorch Fine-Tuning Complete ---")

# --- INFERENCE/GENERATION FUNCTION ---

@torch.no_grad() # No gradients needed for inference
def generate(model, prompt_str, max_new_tokens):
    """Generates text from a prompt (Q|A...)."""
    
    # 1. Prepare initial input tokens
    input_str = prompt_str + "|" 
    idx = [char_to_int.get(c, char_to_int['<PAD>']) for c in input_str]
    idx = torch.tensor(idx, dtype=torch.long, device=DEVICE).unsqueeze(0) # (1, T)

    # 2. Autoregressive Loop
    for _ in range(max_new_tokens):
        # Crop idx to the last BLOCK_SIZE tokens
        idx_cond = idx[:, -BLOCK_SIZE:]
        
        # Get predictions (logits)
        logits, _ = model(idx_cond)
        
        # Focus on the last time step (the predicted next token)
        logits = logits[:, -1, :] # (1, VOCAB_SIZE)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution (multinomial sampling)
        idx_next = torch.multinomial(probs, num_samples=1) # (1, 1)
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) 
        
        # Stop condition: End-of-Sequence token
        if idx_next.item() == char_to_int['\n']:
            break

    # 3. Decode and Clean
    # Convert all generated tokens back to text
    output_tokens = idx[0].tolist()
    text = "".join([int_to_char.get(i, '') for i in output_tokens])
    
    # Find the separator to locate the start of the answer
    try:
        start_index = text.index('|') + 1
    except ValueError:
        start_index = 0
        
    return text[start_index:].replace('\n', '').strip()

# --- Final Demonstration ---
print("\n" + "="*50)
print("✨ PyTorch GPT Answer Generation")
print("="*50)

# The model has been fine-tuned on Q&A data, so it knows the format.
test_prompt = "What is the capital of the United States"

# The model will attempt to generate the answer part after the "|"
answer = generate(model, test_prompt, max_new_tokens=50) 

print(f"Q: {test_prompt}")
print(f"A: {answer}")
print("="*50)
