import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

# Adaptive Learning Memory
class AdaptiveMemory:
    def __init__(self):
        self.preferences = {}  # Stores user-specific adaptations

    def update_preference(self, topic, bot_reply):
        """Updates memory with user preferences dynamically."""
        if topic in self.preferences:
            self.preferences[topic].append(bot_reply)
        else:
            self.preferences[topic] = [bot_reply]

    def get_learned_response(self, topic):
        """Retrieves the learned response for a topic."""
        return self.preferences.get(topic, None)

adaptive_memory = AdaptiveMemory()  # Initialize memory

# Load Model with LoRA Adapter (for lightweight adjustments)
def load_chatbot(model_path="./fine_tuned_ShirayukiV3_OPT"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    # Apply LoRA (Lightweight Fine-Tuning Adapter)
    config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1, 
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, config)

    return model, tokenizer

# Function to fine-tune response preferences in real-time
def adjust_logits(logits, preferred_tokens, tokenizer):
    """Boosts the probability of preferred tokens dynamically."""
    vocab_size = logits.shape[-1]
    mask = torch.zeros(vocab_size).to(logits.device)

    for token in preferred_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        mask[token_id] = 2.0  # Increase probability of preferred words

    logits += mask
    return logits

def generate_dynamic_response(user_input, model, tokenizer):
    """Generates response with adaptive learning."""
    topic = user_input.lower()  # Extract topic

    # Check if bot has learned something about this topic
    learned_response = adaptive_memory.get_learned_response(topic)
    
    # Generate prompt
    prompt = f"<|startoftext|>{user_input}<|sep|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=150, 
            temperature=1.0, 
            top_p=0.9, 
            top_k=50, 
            num_beams=3,
            do_sample=True, 
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("<|sep|>")[-1].strip()

    # If the bot learns something, bias its future responses!
    if learned_response:
        preferred_tokens = learned_response[0].split()[:5]  # Pick key tokens
        logits = adjust_logits(output_ids.logits[:, -1, :], preferred_tokens, tokenizer)
        response = tokenizer.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)

    # Store learning
    adaptive_memory.update_preference(topic, response)

    return response

# Chat Function
def chat(model, tokenizer):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = generate_dynamic_response(user_input, model, tokenizer)

        print("Chatbot:")
        msglist = response.strip().split('<|endoftext|>')
        for sent in msglist:
            print("Sentence:", sent)

if __name__ == "__main__":
    model, tokenizer = load_chatbot()
    chat(model, tokenizer)
