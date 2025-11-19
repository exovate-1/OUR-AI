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
            'to', 'for', 'with', 'about', 'like', 'as', 'if', 'then', 'else',
            'knowledge', 'basic', 'learn', 'teaching', 'answer', 'question'
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
        return " ".join(words) if words else "still learning"
    
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
        
        # HARDCODED API KEY - Your OpenRouter key
        self.api_key = "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"
        print("🔑 OpenRouter API Key: LOADED")
        
        # Initialize with some basic knowledge
        self._initialize_basic_knowledge()
    
    def fetch_knowledge_from_api(self, topic):
        """Use OpenRouter API to get knowledge for training - FIXED VERSION"""
        if not self.api_key:
            print("❌ No API key found")
            return None
            
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/exovate-1/OUR-AI",  # Required by OpenRouter
                "X-Title": "OUR-AI Learning System"  # Required by OpenRouter
            }
            
            # Better prompt for learning
            prompt = f"Explain '{topic}' in simple terms for a beginner. Keep it under 40 words and focus on key concepts."
            
            data = {
                "model": "google/gemma-7b-it:free",  # Using a free model
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 80,
                "temperature": 0.7
            }
            
            print(f"🌐 Calling OpenRouter API for: {topic}")
            response = requests.post(url, headers=headers, json=data, timeout=45)
            
            print(f"📡 API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    print(f"✅ API Success: Received {len(content)} characters")
                    return content.strip()
                else:
                    print(f"❌ API Error: No choices in response")
                    print(f"Response: {result}")
                    return None
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                # Try with a different model if first fails
                return self._fallback_api_call(topic)
                
        except Exception as e:
            print(f"❌ API Request Failed: {e}")
            return None
    
    def _fallback_api_call(self, topic):
        """Fallback API call with different settings"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/exovate-1/OUR-AI",
                "X-Title": "OUR-AI Learning System"
            }
            
            prompt = f"What is {topic}? Explain briefly."
            
            data = {
                "model": "meta-llama/llama-3.1-8b-instruct:free",  # Alternative free model
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 60,
                "temperature": 0.5
            }
            
            print("🔄 Trying fallback API call...")
            response = requests.post(url, headers=headers, json=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"✅ Fallback API Success!")
                return content.strip()
            else:
                print(f"❌ Fallback also failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Fallback API also failed: {e}")
            return None
    
    def learn_from_api(self, topic):
        """Learn about a topic using API and train neural network"""
        print(f"🌐 Learning about '{topic}' from OpenRouter API...")
        
        knowledge = self.fetch_knowledge_from_api(topic)
        if knowledge:
            # Clean the knowledge
            knowledge = self._clean_api_response(knowledge)
            
            # Create Q-A pair and train
            question = f"what is {topic}"
            answer = knowledge
            
            print(f"📚 Training neural network with API knowledge...")
            loss = self.teach(question, answer, epochs=5)
            print(f"✅ Successfully learned about '{topic}' (loss: {loss:.4f})")
            print(f"💡 Knowledge: {knowledge}")
            return True
        else:
            print(f"❌ Failed to learn about '{topic}' from API")
            print("💡 Try a different topic or check your API key")
            return False
    
    def _clean_api_response(self, text):
        """Clean and format API response"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove quotes if present
        text = text.strip('"\'')
        # Limit length
        if len(text) > 200:
            text = text[:200] + "..."
        return text
    
    def _initialize_basic_knowledge(self):
        """Initialize with some basic Q-A pairs"""
        basic_qa = [
            ("hello", "hello how can i help you"),
            ("hi", "hello how are you"),
            ("what is your name", "i am an ai neural network"),
            ("how are you", "i am learning and improving"),
            ("what can you do", "i can learn from conversations and api knowledge"),
            ("who created you", "i was created by exovate using neural networks"),
            ("what is ai", "ai is artificial intelligence machine learning"),
            ("teach me something", "i can learn from api and update my parameters")
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
        print("🧠 TRUE NEURAL NETWORK LLM WITH API LEARNING")
        print("🔑 OpenRouter API: INTEGRATED AND READY")
        print("💾 Knowledge stored in neural parameters through backpropagation")
        print("="*60)
        
        while True:
            print("\nChoose mode:")
            print("1. Teach me (Q -> A)")
            print("2. Ask me something") 
            print("3. Learn from API (Recommended)")
            print("4. Show my brain stats")
            print("5. Save my brain")
            print("6. Load my brain")
            print("7. Test API connection")
            print("8. Exit")
            
            choice = input("\nYour choice (1-8): ").strip()
            
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
                self._test_api()
            elif choice == '8':
                print("👋 Goodbye! All knowledge stored in neural parameters!")
                break
            else:
                print("❌ Invalid choice")
    
    def _test_api(self):
        """Test API connection"""
        print("\n🔍 Testing OpenRouter API connection...")
        test_topic = "artificial intelligence"
        knowledge = self.fetch_knowledge_from_api(test_topic)
        if knowledge:
            print(f"✅ API Connection SUCCESSFUL!")
            print(f"📖 Sample knowledge: {knowledge}")
        else:
            print("❌ API Connection FAILED")
            print("💡 Check your API key or try a different topic")
    
    def _teaching_mode(self):
        """Teaching mode"""
        print("\n🎓 TEACHING MODE")
        print("I will learn by updating my neural weights through backpropagation!")
        
        question = input("Enter the question: ").strip()
        answer = input("Enter the correct answer: ").strip()
        
        if question and answer:
            try:
                epochs = int(input("Training epochs (1-10, default 5): ").strip() or "5")
                epochs = max(1, min(10, epochs))
            except ValueError:
                epochs = 5
                
            self.teach(question, answer, epochs)
            print("✅ Knowledge stored in neural parameters!")
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
            correct = input("Options: 1) Correct me 2) Learn from API 3) Skip: ").strip()
            if correct == '1':
                correct_answer = input("What should I have said? ").strip()
                if correct_answer:
                    self.teach(question, correct_answer)
                    print("✅ Thanks! Neural parameters updated!")
            elif correct == '2':
                topic = input("What related topic should I learn? ").strip()
                if topic:
                    self.learn_from_api(topic)
        else:
            print("❌ Please enter a question")
    
    def _api_learning_mode(self):
        """API-based learning mode"""
        print("\n🌐 API LEARNING MODE")
        print("I will fetch knowledge from OpenRouter and train my neural network!")
        print("Available free models: gemma-7b, llama-3.1-8b")
        
        topic = input("Enter topic to learn: ").strip()
        if topic:
            success = self.learn_from_api(topic)
            if success:
                print("🎉 Success! My neural parameters have been updated with new knowledge!")
            else:
                print("😞 Failed to learn from API. Try manual teaching or different topic.")
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
            print(f"\n📚 Recently learned:")
            for q, a in self.nn.training_pairs[-5:]:
                print(f"   Q: {q}")
    
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

# Simple demonstration
def quick_demo():
    """Quick demonstration"""
    print("🚀 Quick Demo: Neural Network Learning")
    llm = TrueLLM()
    
    # Test basic functionality
    print("\n1. Testing basic knowledge...")
    response = llm.ask("what is your name")
    print(f"Response: {response}")
    
    print("\n2. Testing API learning...")
    llm.learn_from_api("machine learning")
    
    print("\n3. Testing learned knowledge...")
    response = llm.ask("what is machine learning")
    print(f"Response: {response}")

if __name__ == "__main__":
    print("🧠 TRUE NEURAL NETWORK LLM WITH OPENROUTER API")
    print("🔑 API Key: HARDCODED AND READY")
    print("💾 Knowledge stored in W1, W2, b1, b2 parameters")
    
    try:
        mode = input("\nChoose mode:\n1. Quick Demo\n2. Interactive Learning (Recommended)\n3. Test API Only\nChoice (1-3): ").strip()
        
        if mode == '1':
            quick_demo()
        elif mode == '2':
            llm = TrueLLM()
            llm.interactive_learn()
        elif mode == '3':
            llm = TrueLLM()
            llm._test_api()
        else:
            llm = TrueLLM()
            llm.interactive_learn()
            
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔄 Starting in safe mode...")
        llm = TrueLLM()
        llm.interactive_learn()
