
import pandas as pd
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from typing import List, Dict, Tuple

class ContextManager:
    """Manages conversation context and memory for the chatbot."""
    
    def __init__(self, max_history=10):
        self.conversation_history = []
        self.max_history = max_history
        self.personality_traits = {"tsundere_level": 0.8}  # Configurable personality parameters
        
    def add_exchange(self, user_input: str, bot_response: str):
        """Add a conversation exchange to history."""
        self.conversation_history.append({"user": user_input, "bot": bot_response})
        # Keep only recent history to manage context window
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def get_formatted_context(self) -> str:
        """Format conversation history for model input."""
        context = ""
        for exchange in self.conversation_history:
            context += f"<|startoftext|>{exchange['user']}<|sep|>{exchange['bot']}<|endoftext|>"
        return context
    
    def get_context_vector(self, tokenizer, model) -> torch.Tensor:
        """Convert conversation history to vector representation for better understanding."""
        if not self.conversation_history:
            return None
        
        # Generate embeddings for conversation history
        context_text = " ".join([f"{ex['user']} {ex['bot']}" for ex in self.conversation_history])
        inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=512)
        
        # Extract hidden states from model's encoder
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use the average of the last hidden state as context representation
            context_vector = outputs.hidden_states[-1].mean(dim=1)
        
        return context_vector
    
    def save_memory(self, file_path: str):
        """Save conversation history to disk."""
        with open(file_path, 'w') as f:
            json.dump(self.conversation_history, f)
    
    def load_memory(self, file_path: str):
        """Load conversation history from disk."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.conversation_history = json.load(f)

class ConversationDatasetProcessor:
    """Processes raw datasets into structured training data with context."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_path: str) -> Dataset:
        """Load and preprocess conversation data with context awareness."""
        df = pd.read_csv(file_path, encoding="utf-8")
        df['guy'] = df['guy'].str.replace('"', '')
        df['girl'] = df['girl'].str.replace('"', '')
        
        # Generate conversation sequences with context
        processed_data = []
        context_window = []
        
        for i, row in df.iterrows():
            # Add current exchange
            context_window.append((row['guy'], row['girl']))
            
            # Create training example with varying context lengths
            for ctx_size in range(1, min(4, len(context_window) + 1)):
                context = context_window[-ctx_size:]
                
                # Format with context and current exchange
                formatted_sequence = self._format_conversation_sequence(context)
                processed_data.append(formatted_sequence)
            
            # Limit context window size
            if len(context_window) > 5:
                context_window.pop(0)
        
        return Dataset.from_dict({"text": processed_data})
    
    def _format_conversation_sequence(self, context: List[Tuple[str, str]]) -> str:
        """Format a conversation sequence with context."""
        formatted = ""
        # Add context exchanges
        for i in range(len(context) - 1):
            formatted += f"<|startoftext|>{context[i][0]}<|sep|>{context[i][1]}<|endoftext|>"
        
        # Add current exchange that model needs to predict
        current = context[-1]
        formatted += f"<|startoftext|>{current[0]}<|sep|>{current[1]}<|endoftext|>"
        
        return formatted
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset with attention to conversation structure."""
        return dataset.map(
            lambda x: self.tokenizer(
                x['text'], 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_length
            ), 
            batched=True
        )
    
    def get_dynamic_max_length(self, dataset: Dataset, ratio=0.9) -> int:
        """Calculate appropriate max length based on dataset."""
        tokenized_lengths = [len(self.tokenizer(text)["input_ids"]) for text in dataset["text"]]
        max_length = int(max(tokenized_lengths) * ratio)
        return min(max_length, 512)  # Ensuring it doesn't exceed model limits

class TsunderePersonalityModule:
    """Handles the tsundere personality adaptation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Linguistic patterns characteristic of tsundere speech
        self.tsundere_patterns = {
            "denial_phrases": ["I-It's not like I", "Don't get the wrong idea", "I'm not doing this for you"],
            "embarrassment_markers": ["*blushes*", "*looks away*", "Hmph!"],
            "affection_hints": ["Baka", "I guess", "...if you want"],
            "speech_style": ["!?", "...", "Tch."]
        }
        
    def adapt_response(self, response: str, user_input: str, context_vector=None, tsundere_level=0.8) -> str:
        """Adapt a normal response to match tsundere speech patterns."""
        # Skip adaptation if response already has strong tsundere characteristics
        if self._has_tsundere_markers(response):
            return response
            
        # Different adaptation strategies based on the user input and context
        if self._detect_compliment(user_input):
            # High tsundere response for compliments (denial with hidden pleasure)
            response = self._add_tsundere_element(response, "denial_phrases", force=True)
            response = self._add_tsundere_element(response, "embarrassment_markers")
        
        elif self._detect_request(user_input):
            # Reluctant agreement pattern
            if "yes" in response.lower() or "okay" in response.lower() or "sure" in response.lower():
                response = self._add_tsundere_element(response, "affection_hints", force=True)
                
        # General tsundere style adaptations based on level
        if tsundere_level > 0.6:
            response = self._add_tsundere_element(response, "speech_style")
            
        return response
    
    def _has_tsundere_markers(self, text: str) -> bool:
        """Check if the text already contains tsundere markers."""
        for category in self.tsundere_patterns.values():
            for pattern in category:
                if pattern.lower() in text.lower():
                    return True
        return False
    
    def _detect_compliment(self, text: str) -> bool:
        """Detect if user input contains a compliment."""
        compliment_markers = ["nice", "good", "great", "amazing", "beautiful", "pretty", "smart", "cool"]
        return any(marker in text.lower() for marker in compliment_markers)
    
    def _detect_request(self, text: str) -> bool:
        """Detect if user input contains a request."""
        request_markers = ["can you", "could you", "would you", "please", "help me"]
        return any(marker in text.lower() for marker in request_markers)
    
    def _add_tsundere_element(self, text: str, element_type: str, force=False) -> str:
        """Add a tsundere speech element to the text."""
        import random
        
        if not force and random.random() > 0.7:  # 70% chance to add element unless forced
            return text
            
        element = random.choice(self.tsundere_patterns[element_type])
        
        # Different integration strategies based on element type
        if element_type == "denial_phrases":
            if "." in text:
                parts = text.split(".", 1)
                return f"{element}, {parts[0].lower()}. {parts[1]}"
            else:
                return f"{element}, {text.lower()}"
                
        elif element_type == "embarrassment_markers":
            return f"{text} {element}"
            
        elif element_type == "affection_hints":
            if text.endswith("."):
                return f"{text[:-1]}, {element.lower()}."
            else:
                return f"{text}, {element.lower()}"
                
        elif element_type == "speech_style":
            if "!" in text or "?" in text:
                return text
            if text.endswith("."):
                return f"{text[:-1]}{element}"
            else:
                return f"{text}{element}"
        
        return text

class EnhancedChatbot:
    """Main chatbot class integrating all components."""
    
    def __init__(self, model_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.context_manager = ContextManager()
        self.personality = TsunderePersonalityModule(self.tokenizer)
        
        # Special tokens
        if not hasattr(self.tokenizer, 'sep_token'):
            self.tokenizer.add_special_tokens({'sep_token': '<|sep|>'})
        if not hasattr(self.tokenizer, 'bos_token'):
            self.tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
        if not hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        
        # Resize embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def generate_response(self, user_input: str, max_length=100) -> str:
        """Generate a context-aware, personality-consistent response."""
        # Format input with context
        input_text = self._prepare_input_with_context(user_input)
        
        # Generate response
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Get generation parameters based on context
        params = self._get_generation_params(user_input)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_length,
                **params
            )
        
        # Extract only the response part
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = self._extract_response(response)
        
        # Apply personality adaptation
        context_vector = self.context_manager.get_context_vector(self.tokenizer, self.model)
        response = self.personality.adapt_response(
            response, 
            user_input, 
            context_vector,
            self.context_manager.personality_traits["tsundere_level"]
        )
        
        # Update conversation history
        self.context_manager.add_exchange(user_input, response)
        
        return response
    
    def _prepare_input_with_context(self, user_input: str) -> str:
        """Prepare model input with appropriate context."""
        context = self.context_manager.get_formatted_context()
        return f"{context}<|startoftext|>{user_input}<|sep|>"
    
    def _extract_response(self, text: str) -> str:
        """Extract the response part from the generated text."""
        # Find the last separator token and end token
        parts = text.split("<|sep|>")
        if len(parts) > 1:
            response = parts[-1].split("<|endoftext|>")[0].strip()
            return response
        return "I'm not sure what to say..."  # Fallback
    
    def _get_generation_params(self, user_input: str) -> Dict:
        """Get dynamic generation parameters based on input."""
        # Analyze user input for adjusting parameters
        is_question = '?' in user_input
        is_excited = '!' in user_input
        
        # Adjust parameters based on input characteristics
        temperature = 0.9 if is_excited else 0.7
        top_p = 0.92
        repetition_penalty = 1.2
        
        return {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id
        }
    
    def save(self, path: str):
        """Save the model, tokenizer and conversation history."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.context_manager.save_memory(os.path.join(path, "conversation_memory.json"))
    
    def load_memory(self, path: str):
        """Load saved conversation memory."""
        memory_path = os.path.join(path, "conversation_memory.json")
        self.context_manager.load_memory(memory_path)


def fine_tune_enhanced_chat_model(dataset_path, model_name="facebook/opt-350m", 
                                output_dir="./enhanced_tsundere_model"):
    """Fine-tune the OPT model with enhanced context awareness."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if they don't exist
    special_tokens = {'pad_token': tokenizer.eos_token}
    if not hasattr(tokenizer, 'sep_token'):
        special_tokens['sep_token'] = '<|sep|>'
    if not hasattr(tokenizer, 'bos_token'):
        special_tokens['bos_token'] = '<|startoftext|>'
    if not hasattr(tokenizer, 'eos_token'):
        special_tokens['eos_token'] = '<|endoftext|>'
    
    tokenizer.add_special_tokens(special_tokens)
    print("Tokenizer prepared with special tokens")
    
    # Process dataset with context
    processor = ConversationDatasetProcessor(tokenizer)
    dataset = processor.load_data(dataset_path)
    max_length = processor.get_dynamic_max_length(dataset)
    tokenized_dataset = processor.tokenize_data(dataset)
    
    print(f"Dataset processed with context awareness, using max_length={max_length}")
    
    # Load and resize model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure training with focus on context learning
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",  # Changed to evaluate during training
        eval_steps=100,              # Evaluate every 100 steps
        save_strategy="steps",
        save_steps=100,
        num_train_epochs=20,         # Reduced to prevent overfitting
        per_device_train_batch_size=4,  # Smaller batch for deeper learning
        per_device_eval_batch_size=4,
        learning_rate=5e-5,          # Lower learning rate for better generalization
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=3,
        push_to_hub=False,
        # Added gradient accumulation for stable training
        gradient_accumulation_steps=4,
        # Added early stopping
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    # Split dataset for evaluation
    train_size = int(0.9 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
    
    # Setup trainer with evaluation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Added evaluation dataset
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("Starting enhanced training...")
    trainer.train()
    
    print("Saving model, tokenizer, and training artifacts...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir


def demo_chatbot(model_path, num_turns=5):
    """Demonstrate the enhanced chatbot capabilities."""
    print(f"Initializing enhanced chatbot from {model_path}...")
    chatbot = EnhancedChatbot(model_path)
    
    print("Chatbot is ready! Enter your messages (type 'exit' to quit):")
    for _ in range(num_turns):
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = chatbot.generate_response(user_input)
        print(f"Tsundere: {response}")
    
    print("Saving chatbot state...")
    chatbot.save(model_path)
    

if __name__ == "__main__":
    dataset_path = "conversation_dataset_ShirayukiV3.csv"
    output_dir = "./enhanced_tsundere_model"
    
    # Step 1: Fine-tune with enhanced context awareness
    fine_tuned_model_path = fine_tune_enhanced_chat_model(dataset_path, output_dir=output_dir)
    
    # Step 2: Demonstrate the enhanced chatbot
    demo_chatbot(fine_tuned_model_path)