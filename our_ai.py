import numpy as np
import requests
import json
from bs4 import BeautifulSoup

# --- CORRECTED TINYGRAD IMPORTS ---
from tinygrad import Tensor, nn
# Optimizers (like Adam) are in the tinygrad.nn.optim submodule
from tinygrad.nn import optim 
from tinygrad.helpers import getenv 

# Set device to CPU by default, or GPU if available and set
DEVICE = getenv("GPU") 
print(f"Using device: {DEVICE}")

# --- CONFIGURATION (Hyperparameters) ---
BLOCK_SIZE = 128     # Max context length
N_EMBD = 128         # Embedding dimension (d_model)
N_HEAD = 4           # Number of attention heads
N_LAYER = 4          # Number of Transformer blocks
HEAD_SIZE = N_EMBD // N_HEAD
LEARNING_RATE = 1e-4
MAX_ITERS = 100      
BATCH_SIZE = 32
VOCAB_SIZE = 0       

# Data URLs
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
GOOGLE_SEARCH_URL = "https://www.google.com/search?q="
MAX_TRAINING_ENTRIES = 5000

# --- 1. DATA LOADING AND PREPARATION UTILITIES ---
char_to_int = {}
int_to_char = {}

def get_qa_pairs_and_vocab(url, max_entries):
    """Streams data, extracts Q&A, and builds vocabulary."""
    global char_to_int, int_to_char, VOCAB_SIZE
    print(f"-> Streaming JSON data from URL...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    data = response.json()
        
    qa_list = []
    full_text = ""
    
    for document in data.get('data', []):
        for paragraph in document.get('paragraphs', []):
            for qa_pair in paragraph.get('qas', []):
                question = qa_pair.get('question', '')
                answer_text = qa_pair.get('answers', [{}])[0].get('text', '')
                if question and answer_text:
                    qa_list.append((question, answer_text))
                    full_text += question + "|" + answer_text + "\n"
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
    tokens = [char_to_int.get(c, char_to_int['<PAD>']) for c in q] 
    tokens += [char_to_int['|']]
    tokens += [char_to_int.get(c, char_to_int['<PAD>']) for c in a]
    tokens += [char_to_int['\n']]
    
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [char_to_int['<PAD>']] * (max_len - len(tokens))

    return np.array(tokens, dtype=np.int32)

def get_batch(data_tuples, batch_size, block_size):
    """Creates a training batch (X=input, Y=target) from Q&A data."""
    ix = np.random.randint(len(data_tuples), size=batch_size)
    
    X_batch = []
    Y_batch = []
    
    for i in ix:
        q, a = data_tuples[i]
        full_tokens = tokenize_qa(q, a, block_size + 1)
        
        # X: Input (tokens 0 to BLOCK_SIZE-1)
        # Y: Target (tokens 1 to BLOCK_SIZE) - shifted by one
        X_batch.append(full_tokens[:-1])
        Y_batch.append(full_tokens[1:])
        
    # Tensors are created using numpy arrays
    X = Tensor(np.stack(X_batch), device=DEVICE)
    Y = Tensor(np.stack(Y_batch), device=DEVICE)
    return X, Y

# Execute Data Loading
qa_dataset = get_qa_pairs_and_vocab(SQUAD_URL, MAX_TRAINING_ENTRIES)
if not qa_dataset: exit()


# --- 2. TINYGRAD GPT ARCHITECTURE (nn Module) ---
class Head:
    """A single Causal Self-Attention head."""
    def __init__(self, n_embd, head_size):
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        
        # Causal mask creation 
        tril = np.triu(np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32) * -float('inf'), k=1)
        self.tril = Tensor(tril, requires_grad=False, device=DEVICE)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x) 
        q = self.query(x) 
        
        # Compute attention scores: wei = Q @ K^T / sqrt(d_k)
        wei = q.scaled_dot_product_attention(k, transpose=True)
        
        # Causal Masking: prevents looking ahead
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        wei = wei.softmax(axis=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention:
    """Multiple heads running in parallel, then concatenated."""
    def __init__(self, n_embd, n_head):
        self.heads = [Head(n_embd, HEAD_SIZE) for _ in range(n_head)]
        self.proj = nn.Linear(N_EMBD, N_EMBD) 

    def __call__(self, x):
        # Concatenate outputs from all heads
        out = Tensor.cat(*[h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        return out

class FeedFoward:
    """A simple two-layer MLP."""
    def __init__(self, n_embd):
        FF_DIM = n_embd * 4
        self.net = [
            nn.Linear(n_embd, FF_DIM), 
            lambda x: x.relu(),
            nn.Linear(FF_DIM, n_embd),
        ]

    def __call__(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class Block:
    """A single Transformer Block: Attention + FFN with LayerNorm and Residual."""
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_embd, n_head) 
        self.ffwd = FeedFoward(n_embd) 
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        # Pre-Norm + Residual Connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleGPT:
    """The main Generative Pre-trained Transformer model."""
    def __init__(self):
        # Embeddings
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # Stack of Transformer Blocks
        self.blocks = [Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        
        # Final Norm and Head
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        
        # Positional Embeddings
        pos = Tensor.arange(T, requires_grad=False, device=DEVICE) 
        
        # Embeddings: Tokens (B, T, C) + Positions (T, C)
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb 
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # Final Norm and Linear Head
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

        if targets is None:
            return logits, None
        else:
            # Reshape logits and targets for cross_entropy
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            
            # Cross Entropy Loss
            loss = logits_flat.sparse_categorical_crossentropy(targets_flat)
            return logits, loss

# --- 3. TRAINING LOOP ---

# Initialize the Model and Optimizer
model = SimpleGPT()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Optimizer is now imported correctly

print(f"\n--- Starting TinyGrad Q&A Fine-Tuning (Device: {DEVICE}) ---")

# TinyGrad requires using Tensor.train() context for training mode (enables dropout, etc.)
with Tensor.train():
    for iter_num in range(MAX_ITERS):
        # Sample a batch of data
        xb, yb = get_batch(qa_dataset, BATCH_SIZE, BLOCK_SIZE)

        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass and optimization (TinyGrad Autograd in action!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Realize the loss to ensure computation is done and value is fetched
        if iter_num % (MAX_ITERS // 10) == 0 or iter_num == MAX_ITERS - 1:
            print(f"Step {iter_num}/{MAX_ITERS}: Train Loss = {loss.numpy().item():.4f}")

print("--- TinyGrad Fine-Tuning Complete ---")

# --- 4. RAG AND INFERENCE FUNCTIONS ---

def simple_web_search(query):
    """Performs a simple Google search and extracts the snippet content."""
    try:
        search_url = GOOGLE_SEARCH_URL + requests.utils.quote(query)
        print(f"\n[INFO] Searching the web for: '{query}'...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status() 

        soup = BeautifulSoup(response.text, 'html.parser')
        # This class name is a heuristic for Google featured snippets
        snippet_element = soup.find('div', class_='BNeawe s3v9rd AP7Wnd')
        
        if snippet_element:
            snippet = snippet_element.get_text()
            print(f"[INFO] Web context found: {snippet[:100]}...")
            return snippet
        else:
            return "No external context could be retrieved."
            
    except Exception as e:
        print(f"[ERROR] Web search failed: {e}")
        return "No external context could be retrieved from the web."

# Set no_grad for inference to disable gradient computation and save memory/time
@Tensor.no_grad
def generate(model, prompt_str, max_new_tokens):
    """Generates text from a prompt (Q|A...)."""
    
    # 1. Prepare initial input tokens
    input_str = prompt_str + "|" 
    idx = [char_to_int.get(c, char_to_int['<PAD>']) for c in input_str]
    
    # Pad or truncate input sequence to BLOCK_SIZE
    idx = (idx + [char_to_int['<PAD>']] * BLOCK_SIZE)[:BLOCK_SIZE]
    
    # Convert to Tensor (1, T)
    idx = Tensor(np.array(idx, dtype=np.int32), device=DEVICE).unsqueeze(0) 

    # 2. Autoregressive Loop
    for _ in range(max_new_tokens):
        # The input is always the last BLOCK_SIZE tokens
        idx_cond = idx
        
        # Get predictions (logits)
        logits, _ = model(idx_cond)
        
        # Focus on the last time step (the predicted next token)
        logits = logits[:, -1, :] # (1, VOCAB_SIZE)
        
        # Apply softmax to get probabilities
        probs = logits.softmax(axis=-1)
        
        # Sample from the distribution 
        probs_np = probs.numpy().flatten()
        
        # Use np.random.choice for token sampling
        idx_next_np = np.random.choice(VOCAB_SIZE, p=probs_np)
        
        # Update Sequence: Shift the sequence left and append the new token
        idx_np = idx.numpy()
        new_idx_np = np.roll(idx_np[0], -1)
        new_idx_np[-1] = idx_next_np
        idx = Tensor(new_idx_np, device=DEVICE).unsqueeze(0)
        
        # Stop condition: End-of-Sequence token
        if idx_next_np == char_to_int['\n']:
            break

    # 3. Decode and Clean
    output_tokens = idx.numpy()[0].tolist()
    text = "".join([int_to_char.get(i, '') for i in output_tokens])
    
    try:
        start_index = text.index('|') + 1
    except ValueError:
        start_index = 0
        
    return text[start_index:].replace('\n', '').strip()

def rag_with_web_search(model, user_question):
    """The main RAG function combining TinyGrad LLM with Web Search."""
    
    # 1. Search the web for relevant context
    external_context = simple_web_search(user_question)
    
    # 2. Create an augmented prompt for the LLM
    augmented_prompt = (
        f"CONTEXT: {external_context}. "
        f"QUESTION: {user_question}"
    )
    
    # 3. Generate the answer
    final_answer = generate(model, augmented_prompt, max_new_tokens=100)
    
    return final_answer, external_context

# --- Final Demonstration ---

print("\n" + "="*50)
print("✨ TinyGrad GPT RAG Answer Generation")
print("="*50)

# Example question: The model will use the web search context to try and answer
test_prompt = "Who invented the telephone and when was it first demonstrated"

answer, context = rag_with_web_search(model, test_prompt)

print("\n\n" + "="*20 + " RESULTS " + "="*20)
print(f"WEB CONTEXT USED: {context}")
print(f"USER QUESTION: {test_prompt}")
print(f"LLM (RAG) ANSWER: {answer}") 
print("="*50)
