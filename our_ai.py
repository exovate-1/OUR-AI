import numpy as np
import math
import re
import pickle
import os
import requests
import time
from collections import defaultdict

class NeuralNetwork:
    def __init__(self, vocab_size=500, hidden_size=256):
        # Ensure all dimensions match
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size  # Output same size as vocabulary
        
        # Initialize parameters (these will store ALL knowledge)
        self.W1 = np.random.randn(vocab_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, vocab_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, vocab_size))
        
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
            'intelligence', 'neural', 'network', 'code', 'computer', 'science',
            'i', 'am', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'with', 'about', 'like', 'as', 'if', 'then', 'else'
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
        """Convert text to vector representation - fixed size"""
        words = self.preprocess_text(text)
        vector = np.zeros(self.vocab_size)
        for word in words:
            if word in self.vocab:
                idx = self.vocab[word]
                if idx < self.vocab_size:
                    vector[idx] = 1.0
        return vector
    
    def vector_to_text(self, vector, top_k=3):
        """Convert vector back to text"""
        # Get top k words by activation
        top_indices = np.argsort(vector)[-top_k:][::-1]
        words = []
        for idx in top_indices:
            if idx in self.reverse_vocab and vector[idx] > 0.1:
                words.append(self.reverse_vocab[idx])
        return " ".join(words) if words else "learning"
    
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
        
        # Calculate gradients - dimensions are guaranteed to match now
        dZ2 = self.output - y
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
        # Build vocabulary first
        self.build_vocab(question + " " + answer)
        
        # Convert to vectors - guaranteed same size now
        q_vector = self.text_to_vector(question).reshape(1, -1)
        a_vector = self.text_to_vector(answer).reshape(1, -1)
        
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
                # If vocabulary full, we ignore new words
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def generate_response(self, question, creativity=0.2):
        """Generate response using learned parameters"""
        self.build_vocab(question)
        q_vector = self.text_to_vector(question).reshape(1, -1)
        
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
            'hidden_size': self.hidden_size
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
    def __init__(self, vocab_size=500):
        print("🧠 Initializing True Neural Network LLM...")
        self.nn = NeuralNetwork(vocab_size)
        self.conversation_history = []
        self.training_epochs = 0
        self.api_key = "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"
        
        # Initialize with some basic knowledge
        self._initialize_basic_knowledge()
    
    def fetch_knowledge_from_api(self, topic, context=""):
        """Use OpenRouter API to get knowledge for training"""
        if not self.api_key:
            return None
            
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Please provide a concise explanation about '{topic}' {context}.
            Keep it simple and under 50 words. Focus on key concepts.
            """
            
            data = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 100
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                return content.strip()
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ API Request Failed: {e}")
            return None
    
    def learn_from_api(self, topic):
        """Learn about a topic using API and train neural network"""
        print(f"🌐 Learning about '{topic}' from API...")
        
        knowledge = self.fetch_knowledge_from_api(topic)
        if knowledge:
            # Create Q-A pair and train
            question = f"what is {topic}"
            answer = knowledge
            
            loss = self.teach(question, answer, epochs=5)
            print(f"✅ Learned from API about '{topic}' (loss: {loss:.4f})")
            return True
        else:
            print(f"❌ Failed to learn about '{topic}' from API")
            return False
    
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
            print(f"   Learned: '{q}' -> loss: {loss:.4f}")
        print("✅ Basic knowledge initialized!")
    
    def teach(self, question, answer, epochs=5, learning_rate=0.02):
        """Teach the model through parameter updates"""
        print(f"🎓 Teaching: '{question}'")
        
        total_loss = 0
        for epoch in range(epochs):
            loss = self.nn.train_on_pair(question, answer, learning_rate)
            total_loss += loss
            if epoch % 2 == 0:
                print(f"   Epoch {epoch + 1}: loss = {loss:.4f}")
        
        self.training_epochs += epochs
        avg_loss = total_loss / epochs
        
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
        print("🌐 OpenRouter API: ENABLED")
        print("I learn by updating neural parameters through backpropagation!")
        print("="*60)
        
        while True:
            print("\nChoose mode:")
            print("1. Teach me (Q -> A)")
            print("2. Ask me something") 
            print("3. Learn from API")
            print("4. Show my brain stats")
            print("5. Save my brain")
            print("6. Load my brain")
            print("7. Exit")
            
            choice = input("\nYour choice (1-7): ").strip()
            
            if choice == '1':
                self._teaching_mode()
            elif choice == '2':
                self._question_mode()
            elif choice == '3':
                self._api_learning_mode()
            elif choice == '4':
                self._show_stats()
            elif choice == '5':
                self._save_model()
            elif choice == '6':
                self._load_model()
            elif choice == '7':
                print("👋 Goodbye! Knowledge stored in neural parameters!")
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
                epochs = max(1, min(10, epochs))
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
            
            # Ask if they want to correct or learn more
            correct = input("Want to: 1) Correct me 2) Learn more from API 3) Skip: ").strip()
            if correct == '1':
                correct_answer = input("What should I have said? ").strip()
                if correct_answer:
                    self.teach(question, correct_answer)
                    print("✅ Thanks! I've updated my neural parameters!")
            elif correct == '2':
                topic = input("What topic should I learn more about? ").strip()
                if topic:
                    self.learn_from_api(topic)
        else:
            print("❌ Please enter a question")
    
    def _api_learning_mode(self):
        """API-based learning mode"""
        print("\n🌐 API LEARNING MODE")
        print("I will fetch knowledge from OpenRouter and train my neural network!")
        
        topic = input("Enter topic to learn: ").strip()
        if topic:
            self.learn_from_api(topic)
        else:
            print("❌ Please enter a topic")
    
    def _show_stats(self):
        """Show neural network statistics"""
        print(f"\n📊 NEURAL NETWORK STATS:")
        print(f"Training epochs: {self.training_epochs}")
        print(f"Vocabulary size: {len(self.nn.vocab)}")
        print(f"Neural parameters: {self.nn.W1.size + self.nn.W2.size + self.nn.b1.size + self.nn.b2.size:,}")
        print(f"Conversations: {len(self.conversation_history)}")
        print(f"Network architecture: {self.nn.vocab_size} -> {self.nn.hidden_size} -> {self.nn.vocab_size}")
        
        # Show some learned patterns
        if self.nn.training_pairs:
            print(f"\n📚 Recently learned patterns:")
            for q, a in self.nn.training_pairs[-5:]:
                print(f"   '{q}'")
    
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
    ]
    
    for question, answer in teaching_examples:
        llm.teach(question, answer, epochs=3)
        print()
    
    # Test knowledge
    test_questions = [
        "what is python",
        "tell me about machine learning",
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
    print("🧠 TRUE NEURAL NETWORK LLM WITH API LEARNING")
    print("🌐 OpenRouter API: INTEGRATED")
    print("Knowledge is stored in neural weights through backpropagation!")
    
    try:
        mode = input("\nChoose mode:\n1. Demo training\n2. Interactive learning (Recommended)\nChoice (1-2): ").strip()
        
        if mode == '1':
            demonstrate_training()
        else:
            llm = TrueLLM()
            llm.interactive_learn()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Trying to continue in safe mode...")
        llm = TrueLLM()
        llm.interactive_learn()
