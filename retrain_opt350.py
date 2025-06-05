import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")
    df['guy'] = df['guy'].str.replace('"', '')
    df['girl'] = df['girl'].str.replace('"', '')
    data = [f"<|startoftext|>{row['guy']}<|sep|>{row['girl']}<|endoftext|>" for _, row in df.iterrows()]
    return Dataset.from_dict({"text": data})

def tokenize_data(dataset, tokenizer, max_length):
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=max_length), batched=True)

def get_dynamic_max_length(dataset, tokenizer, ratio=0.9):
    tokenized_lengths = [len(tokenizer(text)["input_ids"]) for text in dataset["text"]]
    max_length = int(max(tokenized_lengths) * ratio)
    return min(max_length, 512)

def continue_fine_tuning(dataset_path, base_model_dir="./fine_tuned_ShirayukiV3_OPT", output_dir="./fine_tuned_ShirayukiV4_OPT"):
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")

    dataset = load_data(dataset_path)
    tokenized_dataset = tokenize_data(dataset, tokenizer, get_dynamic_max_length(dataset, tokenizer))
    print("Data tokenized.")

    model = AutoModelForCausalLM.from_pretrained(base_model_dir)
    print("Loaded fine-tuned V3 model.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="epoch",
        num_train_epochs=5,  # Fewer epochs since it's a small update
        per_device_train_batch_size=8,
        warmup_steps=100,
        save_steps=200,
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

    print("Continuing training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model updated and saved.")

if __name__ == "__main__":
    dataset_path = "new_tuples.csv"  # Only the new rows you added
    continue_fine_tuning(dataset_path)
