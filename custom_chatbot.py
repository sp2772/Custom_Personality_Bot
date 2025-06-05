import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_chatbot(model_path="./fine_tuned_ShirayukiV4_OPT"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def chat(model, tokenizer):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        try:
            # temperature = float(input("Enter temperature (0.1 - 1.5): "))
            # top_p = float(input("Enter top-p (0.0 - 1.0): "))
            # top_k = int(input("Enter top-k (1 - 100): "))
            temperature=1.0
            top_p=0.9
            top_k=50
        except ValueError:
            print("Invalid input, using default values.")
            temperature, top_p, top_k = 0.8, 0.9, 50
        
        input_text = f"<|startoftext|>{user_input}<|sep|>"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_length=128, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k, 
                num_beams=3,
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = response.split("<|sep|>")[-1]  # Extract the chatbot's reply
        
        print("Chatbot:")
        msglist = response.strip().split('<|endoftext|>')
        for sent in msglist:
            print("Sentence:", sent)

if __name__ == "__main__":
    model, tokenizer = load_chatbot()
    chat(model, tokenizer)
