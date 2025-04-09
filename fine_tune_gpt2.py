import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")

    data = [f"<|startoftext|>{row['guy']}<|sep|>{row['girl']}<|endoftext|>" for _, row in df.iterrows()]
    return Dataset.from_dict({"text": data})

def tokenize_data(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128), batched=True)

def fine_tune_chat_model(dataset_path, model_name="gpt2", output_dir="./fine_tuned_mode2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_data(dataset_path)
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        push_to_hub=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    dataset_path = "conversation_dataset.csv"  # Update this with the actual path
    fine_tune_chat_model(dataset_path)
 