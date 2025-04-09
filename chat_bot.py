import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_chatbot(model_path="./fine_tuned_mode2"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def chat(model, tokenizer):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        input_text = f"<|startoftext|>{user_input}<|sep|>"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response.split("<|sep|>")[-1]  # Extract the chatbot's reply
        print(f"Chatbot: {response.strip()}")
        

if __name__ == "__main__":
    model, tokenizer = load_chatbot()
    chat(model, tokenizer)
