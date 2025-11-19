import numpy as np
import math
import re
import pickle
import os
from collections import defaultdict
import time

class NeuralNetwork:
    def __init__(self, vocab_size, hidden_size=256, output_size=256):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize parameters (these will store ALL knowledge)
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.01  # Input to hidden
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Hidden to output
        self.b1 = np.zeros((1, hidden_size))  # Hidden bias
        self.b2 = np.zeros((1, output_size))  # Output bias
        
        # Training memory
        self.training_pairs = []
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.reverse_vocab = {}
        
        # Initialize vocabulary with some basic words
        self._initialize_basic_vocab()
    
    def _initialize_basic_vocab(self):
        """Initialize with common words to ensure vocabulary size"""
        basic_words = [
            'hello', 'hi', 'what', 'is', 'your', 'name', 'how', 'are', 'you',
            'can', 'do', 'help', 'thank', 'thanks', 'bye', 'goodbye', 'yes', 'no',
            'python', 'programming', 'machine', 'learning', 'ai', 'artificial',
            'intelligence', 'neural', 'network', 'code', 'computer', 'science'
        ]
        for word in basic_words:
            self.vocab[word]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def preprocess_text(self, text):
        """Tokenize text"""
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def text_to_vector(self, text):
        """Convert text to vector representation - ensure fixed size"""
        words = self.preprocess_text(text)
        vector = np.zeros(self.vocab_size)
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                if idx < self.vocab_size:  # Ensure within bounds
                    vector[idx] = 1.0
        return vector
    
    def vector_to_text(self, vector, threshold=0.1):
        """Convert vector back to text"""
        words = []
        for idx, value in enumerate(vector):
            if value > threshold and idx in self.reverse_vocab:
                words.append(self.reverse_vocab[idx])
        return " ".join(words) if words else "i don't know enough yet"
    
    def forward(self, x):
        """Forward pass through the network"""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # Hidden layer activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = np.tanh(self.z2)  # Output layer activation
        return self.output
    
    def backward(self, x, y, learning_rate=0.01):
        """Backward pass - update parameters (THIS IS WHERE LEARNING HAPPENS)"""
        m = x.shape[0]
        
        # Ensure dimensions match
        if self.output.shape != y.shape:
            # If shapes don't match, we need to handle this gracefully
            min_cols = min(self.output.shape[1], y.shape[1])
            output_trimmed = self.output[:, :min_cols]
            y_trimmed = y[:, :min_cols]
            dZ2 = output_trimmed - y_trimmed
        else:
            dZ2 = self.output - y
        
        # Calculate gradients
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (1 - np.power(self.a1, 2))  # tanh derivative
        dW1 = (1/m) * np.dot(x.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update parameters (ACTUAL LEARNING)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train_on_pair(self, question, answer, learning_rate=0.01):
        """Train the network on a single Q-A pair"""
        # Ensure vocabulary is built
        self.build_vocab(question + " " + answer)
        
        # Convert to vectors
        q_vector = self.text_to_vector(question).reshape(1, -1)
        a_vector = self.text_to_vector(answer).reshape(1, -1)
        
        # Ensure vectors are the right size
        if q_vector.shape[1] > self.vocab_size:
            q_vector = q_vector[:, :self.vocab_size]
        if a_vector.shape[1] > self.vocab_size:
            a_vector = a_vector[:, :self.vocab_size]
        
        # Pad if necessary
        if q_vector.shape[1] < self.vocab_size:
            q_vector = np.pad(q_vector, ((0,0), (0, self.vocab_size - q_vector.shape[1])))
        if a_vector.shape[1] < self.vocab_size:
            a_vector = np.pad(a_vector, ((0,0), (0, self.vocab_size - a_vector.shape[1])))
        
        # Forward pass
        output = self.forward(q_vector)
        
        # Backward pass - update parameters
        self.backward(q_vector, a_vector, learning_rate)
        
        # Store training pair
        self.training_pairs.append((question, answer))
        
        return np.mean(np.square(output - a_vector))  # Return loss
    
    def build_vocab(self, text):
        """Build vocabulary from text"""
        words = self.preprocess_text(text)
        for word in words:
            if word not in self.vocab:
                if len(self.vocab) < self.vocab_size:
                    self.vocab[word]
                else:
                    # Vocabulary full, replace least used word or ignore
                    pass
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def generate_response(self, question, creativity=0.1):
        """Generate response using learned parameters"""
        self.build_vocab(question)
        q_vector = self.text_to_vector(question).reshape(1, -1)
        
        # Ensure vector is correct size
        if q_vector.shape[1] > self.vocab_size:
            q_vector = q_vector[:, :self.vocab_size]
        if q_vector.shape[1] < self.vocab_size:
            q_vector = np.pad(q_vector, ((0,0), (0, self.vocab_size - q_vector.shape[1])))
        
        # Forward pass through learned parameters
        response_vector = self.forward(q_vector)
        
        # Add some creativity (noise)
        response_vector += creativity * np.random.randn(*response_vector.shape)
        
        return self.vector_to_text(response_vector[0])
    
    def save_model(self, filename):
        """Save the actual neural network parameters"""
        model_data = {
            'W1': self.W1,
            'W2': self.W2,
            'b1': self.b1,
            'b2': self.b2,
            'vocab': dict(self.vocab),
            'reverse_vocab': self.reverse_vocab,
            'training_pairs': self.training_pairs,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"🧠 Model parameters saved to {filename}")
    
    def load_model(self, filename):
        """Load neural network parameters"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.W1 = model_data['W1']
            self.W2 = model_data['W2']
            self.b1 = model_data['b1']
            self.b2 = model_data['b2']
            self.vocab = defaultdict(lambda: len(self.vocab), model_data['vocab'])
            self.reverse_vocab = model_data['reverse_vocab']
            self.training_pairs = model_data['training_pairs']
            print(f"🧠 Model parameters loaded from {filename}")
            return True
        else:
            print(f"❌ No saved model found at {filename}")
            return False

class TrueLLM:
    def __init__(self, vocab_size=500):  # Reduced vocab size for stability
        print("🧠 Initializing True Neural Network LLM...")
        self.nn = NeuralNetwork(vocab_size)
        self.conversation_history = []
        self.training_epochs = 0
        
        # Initialize with some basic knowledge
        self._initialize_basic_knowledge()
    
    def _initialize_basic_knowledge(self):
        """Initialize with some basic Q-A pairs"""
        basic_qa = [
            ("hello", "hello how can i help you"),
            ("hi", "hello how are you"),
            ("what is your name", "i am an ai assistant"),
            ("how are you", "i am functioning well thank you"),
            ("what can you do", "i can learn from our conversations and help you")
        ]
        
        print("📚 Learning basic knowledge...")
        for q, a in basic_qa:
            loss = self.nn.train_on_pair(q, a, learning_rate=0.05)
            print(f"   Learned: '{q}' -> '{a}' (loss: {loss:.4f})")
        print("✅ Basic knowledge initialized!")
    
    def teach(self, question, answer, epochs=5, learning_rate=0.02):
        """Teach the model through parameter updates"""
        print(f"🎓 Teaching: '{question}' -> '{answer}'")
        
        total_loss = 0
        for epoch in range(epochs):
            loss = self.nn.train_on_pair(question, answer, learning_rate)
            total_loss += loss
            if epoch % 2 == 0:  # Print every 2 epochs to reduce spam
                print(f"   Epoch {epoch + 1}: loss = {loss:.4f}")
        
        self.training_epochs += epochs
        avg_loss = total_loss / epochs
        print(f"✅ Learned! Average loss: {avg_loss:.4f}")
        
        # Store conversation
        self.conversation_history.append({
            'type': 'teaching',
            'question': question,
            'answer': answer,
            'epochs': epochs,
            'loss': avg_loss,
            'timestamp': time.time()
        })
        
        return avg_loss
    
    def ask(self, question):
        """Ask question and get response from neural network"""
        print(f"❓ You asked: '{question}'")
        
        # Generate response from neural network parameters
        response = self.nn.generate_response(question)
        
        # Store conversation
        self.conversation_history.append({
            'type': 'question',
            'question': question,
            'response': response,
            'timestamp': time.time()
        })
        
        return response
    
    def interactive_learn(self):
        """Interactive learning session"""
        print("\n" + "="*60)
        print("🧠 TRUE NEURAL NETWORK TRAINING MODE")
        print("I learn by updating my neural network parameters through backpropagation!")
        print("Knowledge is stored in weights (W1, W2, b1, b2), not in databases!")
        print("="*60)
        
        while True:
            print("\nChoose mode:")
            print("1. Teach me (Q -> A)")
            print("2. Ask me something")
            print("3. Show my brain stats")
            print("4. Save my brain")
            print("5. Load my brain") 
            print("6. Exit")
            
            choice = input("\nYour choice (1-6): ").strip()
            
            if choice == '1':
                self._teaching_mode()
            elif choice == '2':
                self._question_mode()
            elif choice == '3':
                self._show_stats()
            elif choice == '4':
                self._save_model()
            elif choice == '5':
                self._load_model()
            elif choice == '6':
                print("👋 Goodbye! Remember, all knowledge is in my neural parameters!")
                break
            else:
                print("❌ Invalid choice")
    
    def _teaching_mode(self):
        """Teaching mode"""
        print("\n🎓 TEACHING MODE")
        print("I will learn by updating my weights through backpropagation!")
        
        question = input("Enter the question: ").strip()
        answer = input("Enter the correct answer: ").strip()
        
        if question and answer:
            try:
                epochs = int(input("Training epochs (1-10, default 5): ").strip() or "5")
                epochs = max(1, min(10, epochs))  # Limit between 1-10
            except ValueError:
                epochs = 5
                
            self.teach(question, answer, epochs)
        else:
            print("❌ Please provide both question and answer")
    
    def _question_mode(self):
        """Question answering mode"""
        print("\n❓ QUESTION MODE")
        question = input("Ask me something: ").strip()
        if question:
            response = self.ask(question)
            print(f"🧠 My response: {response}")
            
            # Ask if they want to correct the response
            correct = input("Is this correct? (y/n): ").strip().lower()
            if correct == 'n':
                correct_answer = input("What should I have said? ").strip()
                if correct_answer:
                    self.teach(question, correct_answer)
                    print("✅ Thanks! I've updated my neural parameters!")
        else:
            print("❌ Please enter a question")
    
    def _show_stats(self):
        """Show neural network statistics"""
        print(f"\n📊 NEURAL NETWORK STATS:")
        print(f"Training epochs: {self.training_epochs}")
        print(f"Vocabulary size: {len(self.nn.vocab)}")
        print(f"Neural parameters: {self.nn.W1.size + self.nn.W2.size + self.nn.b1.size + self.nn.b2.size:,}")
        print(f"Conversations: {len(self.conversation_history)}")
        print(f"Network architecture: {self.nn.vocab_size} -> {self.nn.hidden_size} -> {self.nn.output_size}")
        
        # Show some learned patterns
        if self.nn.training_pairs:
            print(f"\n📚 Recently learned patterns:")
            for q, a in self.nn.training_pairs[-5:]:
                print(f"   '{q}' -> '{a}'")
    
    def _save_model(self):
        """Save the model parameters"""
        filename = input("Enter filename to save (default: brain.pkl): ").strip() or "brain.pkl"
        self.nn.save_model(filename)
    
    def _load_model(self):
        """Load model parameters"""
        filename = input("Enter filename to load (default: brain.pkl): ").strip() or "brain.pkl"
        if self.nn.load_model(filename):
            # Update conversation history from loaded pairs
            self.conversation_history = []
            for q, a in self.nn.training_pairs:
                self.conversation_history.append({
                    'type': 'teaching',
                    'question': q,
                    'answer': a,
                    'timestamp': time.time()
                })

def demonstrate_training():
    """Demonstrate the neural network learning"""
    print("🧪 DEMONSTRATING NEURAL NETWORK LEARNING")
    print("="*50)
    
    # Create a fresh LLM
    llm = TrueLLM()
    
    # Teach some concepts
    teaching_examples = [
        ("what is python", "python is a programming language"),
        ("what is machine learning", "machine learning is a type of artificial intelligence"),
        ("how to learn programming", "start with basic concepts and practice regularly"),
    ]
    
    for question, answer in teaching_examples:
        llm.teach(question, answer, epochs=3)
        print()
    
    # Test knowledge
    test_questions = [
        "what is python",
        "tell me about machine learning", 
        "how to learn programming",
    ]
    
    print("🧠 TESTING LEARNED KNOWLEDGE")
    print("="*50)
    for question in test_questions:
        response = llm.ask(question)
        print(f"Q: {question}")
        print(f"A: {response}")
        print()
    
    # Show final stats
    llm._show_stats()

if __name__ == "__main__":
    print("🧠 TRUE NEURAL NETWORK LLM")
    print("This AI learns by updating neural network parameters through backpropagation!")
    print("Knowledge is stored in weights (W1, W2, b1, b2), not in databases!")
    
    try:
        mode = input("\nChoose mode:\n1. Demo training\n2. Interactive learning\nChoice (1-2): ").strip()
        
        if mode == '1':
            demonstrate_training()
        else:
            llm = TrueLLM()
            llm.interactive_learn()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("If you're having issues, try running in interactive mode (option 2) first.")
