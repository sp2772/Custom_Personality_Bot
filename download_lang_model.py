
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import multiprocessing
def test_translate(model, tokenizer):
    try:
        # Example: Translate from English to French
        text = "Hello, how are you?"
        
        # Set source language
        tokenizer.src_lang = "en"
        
        # Encode the text
        encoded = tokenizer(text, return_tensors="pt")
        
        # Generate translation
        tokenizer.tgt_lang = "fr"
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id("fr"))
        
        # Decode the generated tokens
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print(f"English: {text}")
        print(f"French: {translation}")
    except Exception as e:
        print(f"Error testing translation: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    model_name = "facebook/m2m100_418M"
    cache_dir = "./model_cache"  # Local storage directory

    # Download model and tokenizer
    tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=True)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=True, use_safetensors=False)
    
    print("Model downloaded successfully and saved in:", cache_dir)
    
    # Optional: Test the model with a simple translation to verify it works
    print("\nTesting model with a simple translation:")
    test_translate(model, tokenizer)