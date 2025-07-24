import discord
from discord.ext import commands, tasks
import requests
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import re
import time
import random
import asyncio
import threading
import os
import shutil
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import queue
import logging
from collections import defaultdict
from datasets import disable_progress_bar
#disable_progress_bar()
# 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Discord bot token and channel ID - DEFINE YOUR VALUES HERE
TOKEN = "<bot token>"
PROGRESS_CHANNEL_ID = None  # Set to your channel ID or None to disable Discord progress messages

# Global variables
s_model = SentenceTransformer("all-MiniLM-L6-v2")
frac_of_new_dataset = 1
training_queue = queue.Queue()
training_in_progress = False
current_model_number = 0
training_progress = {"status": "idle", "epoch": 0, "total_epochs": 0}
existing_models = set()  # Track which models exist

# File paths
CSV_FILE = "new_tuples.csv" # This is the CSV file for training data 
MODELS_DIR = "models"
STATE_FILE = "bot_state.json"
PREV_DATA_PATH = "conversation_dataset_ShirayukiV3.csv" 
BASE_MODEL_DIR = "./fine_tuned_ShirayukiV5_OPT"  # Base model directory

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)


def input_clean_func(input_string):
    """Clean Discord mentions from input string while keeping identity awareness"""
    self_id = "1355213169920049364"
    mention_pattern = r'(<?@?\d+>?)'

    def replace_mention(match):
        raw = match.group()
        uid_match = re.search(r'\d+', raw)
        if not uid_match:
            return raw  # Return original if no digits found
        uid = uid_match.group()
        return "you" if uid == self_id else "someone"

    cleaned_input = re.sub(mention_pattern, replace_mention, input_string)
    return cleaned_input

def load_chatbot(model_path="./fine_tuned_ShirayukiV3_OPT"):
    """Load chatbot model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def get_Shirayuki_response(model, tokenizer, user_input, max_seq=128, temp=1.0, top_p=0.9, top_k=50):
    """Generate response from Shirayuki model"""
    input_text = f"<|startoftext|>{user_input}<|sep|>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=max_seq, 
            temperature=temp, 
            do_sample=True,
            top_p=top_p, 
            top_k=top_k, 
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response.split("<|sep|>")[-1]  
    return response.strip()

def discover_existing_models():
    """Discover which models exist in the filesystem"""
    global existing_models
    existing_models = set()
    
    # Check base model
    if os.path.exists(BASE_MODEL_DIR):
        existing_models.add(0)
    
    # Check numbered models
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            if item.startswith("model") and os.path.isdir(os.path.join(MODELS_DIR, item)):
                try:
                    model_num = int(item.replace("model", ""))
                    existing_models.add(model_num)
                except ValueError:
                    continue
    
    print(f"[Discovery] Found existing models: {sorted(existing_models)}")

def save_state():
    """Save bot state to file"""
    state = {
        "current_model_number": current_model_number,
        "training_in_progress": training_in_progress,
        "existing_models": list(existing_models)
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state():
    """Load bot state from file"""
    global current_model_number, training_in_progress, existing_models
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            current_model_number = state.get("current_model_number", 0)
            training_in_progress = state.get("training_in_progress", False)
            existing_models = set(state.get("existing_models", []))
    
    # Always discover models from filesystem as backup
    discover_existing_models()
    
    # Reconcile state with filesystem
    if current_model_number not in existing_models and existing_models:
        current_model_number = max(existing_models)
        print(f"[State] Reconciled current_model_number to {current_model_number}")

def get_bot_model_path():
    """Get path to model that bot should use (N-1)"""
    # Bot uses the model before the current one being trained
    if current_model_number <= 1:
        return BASE_MODEL_DIR
    
    # Use model N-1 (previous model)
    bot_model_number = current_model_number - 1
    bot_model_path = os.path.join(MODELS_DIR, f"model{bot_model_number}")
    
    # Fallback to base model if bot model doesn't exist
    if not os.path.exists(bot_model_path):
        print(f"[Bot Model] Model {bot_model_number} not found, using base model")
        return BASE_MODEL_DIR
    
    return bot_model_path

def get_training_model_path():
    """Get path to model for training (N)"""
    # Training uses the current model number
    if current_model_number == 0:
        return BASE_MODEL_DIR
    return os.path.join(MODELS_DIR, f"model{current_model_number}")

def get_next_model_path():
    """Get path for next model (N+1)"""
    return os.path.join(MODELS_DIR, f"model{current_model_number + 1}")

def initialize_csv():
    """Initialize CSV file if it doesn't exist"""
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['guy', 'girl'])
        df.to_csv(CSV_FILE, index=False)

def add_to_csv(user_input, correct_response):
    """Add correction to CSV file"""
    try:
        df = pd.read_csv(CSV_FILE)
        new_row = pd.DataFrame({'guy': [user_input], 'girl': [correct_response]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"[CSV] Added correction: {user_input} -> {correct_response}")
        return len(df)
    except Exception as e:
        print(f"[CSV Error] {e}")
        return 0

def get_csv_row_count():
    """Get number of rows in CSV file"""
    try:
        df = pd.read_csv(CSV_FILE)
        return len(df)
    except:
        return 0

def remove_trained_rows(num_rows):
    """Remove specified number of rows from beginning of CSV and back them up to a separate file"""
    try:
        df = pd.read_csv(CSV_FILE)
        if len(df) >= num_rows:
            removed_df = df.iloc[:num_rows]
            df = df.iloc[num_rows:]

            # Save removed rows to backup CSV
            backup_file = "trained_rows_backup.csv"
            if os.path.exists(backup_file):
                removed_df.to_csv(backup_file, mode='a', index=False, header=False)
            else:
                removed_df.to_csv(backup_file, index=False)
            df.to_csv(CSV_FILE, index=False)
            print(f"[CSV] Removed {num_rows} trained rows and backed them up to '{backup_file}'")
            
    except Exception as e:
        print(f"[CSV Error] {e}")


# Fine-tuning functions (adapted from your code)
def load_data(file_path):
    """Load data from CSV file"""
    print(f"[Loading] Reading file: {file_path}")
    df = pd.read_csv(file_path, encoding="utf-8")
    df['guy'] = df['guy'].str.replace('"', '')
    df['girl'] = df['girl'].str.replace('"', '')
    print(f"[Loaded] {len(df)} rows from {file_path}")
    return df

def format_dataset(df):
    """Format dataset for training"""
    print(f"[Formatting] Creating text format dataset...")
    data = [f"<|startoftext|>{row['guy']}<|sep|>{row['girl']}<|endoftext|>" for _, row in df.iterrows()]
    return Dataset.from_dict({"text": data})

def tokenize_data(dataset, tokenizer, max_length):
    """Tokenize dataset"""
    print(f"[Tokenizing] With max_length={max_length}")
    return dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding="max_length", max_length=max_length), batched=True)

def get_dynamic_max_length(dataset, tokenizer, ratio=0.9):
    """Get dynamic max length from dataset"""
    print("[Computing] Dynamic max token length from dataset...")
    tokenized_lengths = [len(tokenizer(text)["input_ids"]) for text in dataset["text"]]
    length = min(int(max(tokenized_lengths) * ratio), 512)
    print(f"[Computed] Max length set to {length}")
    return length

def generate_variations(sentence):
    """Generate variations of a sentence"""
    variations = [sentence]
    
    if " " in sentence:
        words = sentence.split()
        idx = random.randint(0, len(words)-1)
        fillers = ["uh", "like", "well", "so", "you know", "I mean", "basically", "actually"]
        words.insert(idx, random.choice(fillers))
        variations.append(" ".join(words))

    if len(sentence.split()) > 3:
        words = sentence.split()
        for i in range(len(words)):
            if random.random() < 0.2:
                j = random.randint(0, len(words)-1)
                words[i], words[j] = words[j], words[i]
        variations.append(" ".join(words))

    variations.append(f"{sentence}, you know?")
    return variations[:3]

def evaluate_model(model, tokenizer, eval_df, reference_df, max_len=512, threshold=0.78,real_threshold=0.70, variation_test=True, guy_similarity_threshold=0.75):
    """Evaluate model performance"""
    eval_guy_inputs = eval_df['guy'].unique()
    correct = 0
    total = len(eval_guy_inputs)
    pbar=tqdm(eval_guy_inputs, total=total, desc="[Evaluating]",leave=False)
    for guy in pbar:
        guy_length = len(guy)
        min_length = int(guy_length * 0.7)
        max_length = int(guy_length * 1.3)
        
        filtered_ref_df = reference_df[
            (reference_df['guy'].str.len() >= min_length) & 
            (reference_df['guy'].str.len() <= max_length)
        ].copy()
        print(f"[Filtering] '{guy}' -> {len(filtered_ref_df)} reference rows")
        girl_responses = []
        eval_guy_emb = s_model.encode(guy, convert_to_tensor=True,show_progress_bar=False)
        
        try: 
            for i, (_, ref_row) in enumerate(filtered_ref_df.iterrows()):
                ref_guy = ref_row['guy']
                ref_guy_emb = s_model.encode(ref_guy, convert_to_tensor=True,show_progress_bar=False)
                similarity = util.cos_sim(eval_guy_emb, ref_guy_emb).item()
                if similarity >= guy_similarity_threshold:
                    girl_responses.append(ref_row['girl'])
        except Exception as e:
            print(f"[LOOP Error] {e}")
            continue

        girl_responses = list(dict.fromkeys(girl_responses))
        print(f"[Collected] {len(girl_responses)} responses for '{guy}'")
        if not girl_responses:
            continue

        prompts = generate_variations(guy) if variation_test else [guy]
        matched = False
        #print("prob3")
        for prompt in prompts:
            input_text = f"<|startoftext|>{prompt}<|sep|>"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                input_len = inputs['input_ids'].shape[1]
                max_total_len = max_len
                max_new_tokens = max(75, max_total_len - input_len)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                outputs2 = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=1.0, top_p=0.9, top_k=50
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

            try:
                if "<|sep|>" in decoded and "<|endoftext|>" in decoded:
                    response = decoded.split("<|sep|>")[-1].strip().split("<|endoftext|>")[0].strip()
                elif "<|sep|>" in decoded:
                    response = decoded.split("<|sep|>")[-1].strip()
                elif "<|endoftext|>" in decoded:
                    response = decoded.split("<|endoftext|>")[0].strip()
                else:
                    response = decoded.strip()
            except:
                response = ""
                
            try:
                if "<|sep|>" in decoded2 and "<|endoftext|>" in decoded2:
                    response2 = decoded2.split("<|sep|>")[-1].strip().split("<|endoftext|>")[0].strip()
                elif "<|sep|>" in decoded2:
                    response2 = decoded2.split("<|sep|>")[-1].strip()
                elif "<|endoftext|>" in decoded2:
                    response2 = decoded2.split("<|endoftext|>")[0].strip()
                else:
                    response2 = decoded2.strip()
            except:
                response2 = ""
                    
            
            emb1 = s_model.encode(response, convert_to_tensor=True,show_progress_bar=False)
            emb3 = s_model.encode(response2, convert_to_tensor=True,show_progress_bar=False)
            best_sim = -1.0
            model_len = len(response2)

            for target in girl_responses:
                target_len= len(target)
                if not (0.6 * model_len <= target_len <= 1.4 * model_len):
                    print(f"[Skip] Real Length mismatch: target='{target[:10]}' (len={target_len}) not in range {int(0.6 * model_len)}–{int(1.4 * model_len)} of model response '{response[:10]}' (len={model_len})")
                    continue
                emb2 = s_model.encode(target, convert_to_tensor=True,show_progress_bar=False)
                sim = util.cos_sim(emb1, emb2).item()
                sim2 = util.cos_sim(emb3, emb2).item()
                best_sim = max(best_sim, sim)

                if  sim2 >= real_threshold or sim >= threshold:
                    res_str = "real threshold sim:"+str(sim2) if sim2 >= real_threshold else "exceeding ideal threshold"
                    print(f"[Match] '{guy[:10]}' matched with response: '{response[:10]}' and '{response2[:10]}' \n(ideal sim={sim:.2f}), because of {res_str}")
                    matched = True
                    break

            if matched:
                break

        if matched:
            correct += 1
    pbar.close()
    #print("prob4")
    acc = correct / total
    print(f"\n[Evaluation Complete] Accuracy: {correct}/{total} = {acc:.2%}")
    return acc

def perform_training():
    """Perform model training"""
    global current_model_number, training_in_progress, training_progress, existing_models
    
    try:
        training_in_progress = True
        training_progress["status"] = "starting"
        next_model_num = current_model_number + 1
        print(f"[Training] Starting training for model{next_model_num}")
        
        # Load data
        new_df = load_data(CSV_FILE)
        if len(new_df) < 5:
            print("[Training] Not enough data to train")
            return
        
        # Take first 5 rows for training
        training_df = new_df.head(5)
        #remove_trained_rows(5)
        
        prev_df = load_data(PREV_DATA_PATH)
        ref_df = pd.concat([training_df, prev_df]).drop_duplicates().sample(frac=1).reset_index(drop=True)
        
        # Load current training model (model N)
        training_model_path = get_training_model_path()
        print(f"[Training] Loading training model from: {training_model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(training_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(training_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        
        max_len = get_dynamic_max_length(format_dataset(training_df), tokenizer)
        
        # Training parameters
        patience = 6
        max_epochs = 30
        new_threshold = 0.79
        old_threshold = 0.74
        prev_sample_size_train = 50
        prev_sample_size_eval = 8
        
        best_epoch = -1
        patience_counter = 0
        
        training_progress["total_epochs"] = max_epochs
        
        for epoch in range(max_epochs):
            training_progress["epoch"] = epoch + 1
            training_progress["status"] = f"training_epoch_{epoch + 1}"
            
            print(f"\n===== Training Epoch {epoch + 1}/{max_epochs} =====")
            
            sampled_prev_df = prev_df.sample(n=prev_sample_size_train,random_state = random.Random(int(time.time() * 1e6)).randint(2, 2**30 - 1))
            mixed_df = pd.concat([training_df, sampled_prev_df]).sample(frac=1).reset_index(drop=True)
            print(f"[Mixing] Mixed dataset size: {len(mixed_df)}")
            tokenized_dataset = tokenize_data(format_dataset(mixed_df), tokenizer, max_len)
            
            output_dir = get_next_model_path()
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=8,
                num_train_epochs=2,
                save_strategy="no",
                logging_dir="./logs",
                logging_steps=10
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
            )
            
            print("[Training] Starting training for 1 epoch...")
            trainer.train()
            print("[Training] Done.")
            
            # Evaluation
            print("[Evaluation] Evaluating on new and old datasets...")
            
            eval_prev_df = prev_df.sample(n=prev_sample_size_eval,random_state = random.Random(int(time.time() * 1e6)).randint(2, 2**30 - 1))
            if len(training_df) == 0 or len(eval_prev_df) == 0:
                print("No data for evaluation, skipping evaluation.")
                break
            new_acc = evaluate_model(model, tokenizer, training_df, ref_df, max_len=max_len,threshold=0.75)
            old_acc = evaluate_model(model, tokenizer, eval_prev_df, ref_df, max_len=max_len,threshold=0.70)
            
            print(f"[Eval] New acc: {new_acc:.2f}, Old acc: {old_acc:.2f}")
            
            if new_acc >= new_threshold and old_acc >= old_threshold:
                print("✅ Early stopping: Model satisfies both requirements.")
                best_epoch = epoch
                break
            
            if new_acc >=new_threshold and patience_counter < patience-2:
                print(f"✅ New model meets new threshold: {new_acc:.2f} >= {new_threshold:.2f} under patience {patience_counter}/{patience}")
                best_epoch = epoch
                patience_counter = 0
                print("Early stopping, attained required model performance.")
                break
            
            if old_acc < old_threshold:
                patience_counter += 1
                print(f"❌ Patience counter: {patience_counter}/{patience}")
            else:
                patience_counter = 0
                print(f"✅ Resetting patience counter.")
                
            if patience_counter >= patience:
                print("⏹️ Early stopping: Patience exceeded.")
                break
            
        
        # Save the new model
        next_model_path = get_next_model_path()
        print(f"[Saving] Saving model to: {next_model_path}")
        model.save_pretrained(next_model_path)
        tokenizer.save_pretrained(next_model_path)
        
        # Update model management
        current_model_number += 1
        existing_models.add(current_model_number)
        
        # Delete old model if we have more than 3 models (keep N-2, N-1, N)
        if current_model_number > 3:
            # Delete model N-2 (two models behind current)
            old_model_num = current_model_number - 2
            old_model_path = os.path.join(MODELS_DIR, f"model{old_model_num}")
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
                existing_models.discard(old_model_num)
                print(f"[Cleanup] Deleted old model{old_model_num}")
        
        # Remove trained rows from CSV
        remove_trained_rows(5)
        
        training_progress["status"] = "completed"
        print(f"[Training] Completed training for model{current_model_number}")
        
    except Exception as e:
        print(f"[Training Error] {e}")
        training_progress["status"] = "error"
    finally:
        training_in_progress = False
        save_state()

def training_worker():
    """Background training worker"""
    while True:
        try:
            # Check if training is needed
            if not training_in_progress and get_csv_row_count() >= 5:
                perform_training()
                # Reload model in Discord bot
                asyncio.run_coroutine_threadsafe(reload_model(), bot.loop)
            
            time.sleep(10)  # Check every 10 seconds
        except Exception as e:
            print(f"[Training Worker Error] {e}")
            time.sleep(30)

class CorrectionView(discord.ui.View):
    """Discord view for correction button"""
    
    def __init__(self, original_message, bot_response):
        super().__init__(timeout=300)
        self.original_message = original_message
        self.bot_response = bot_response
        self.correction_received = False
    
    @discord.ui.button(label='Correct Response', style=discord.ButtonStyle.primary, emoji='✏️')
    async def correct_response(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.correction_received:
            await interaction.response.send_message("Correction already received for this message.", ephemeral=True)
            return
        
    
        await interaction.response.send_message(
            f"H-Hey! I’m your newlywed Tsundere wife… not that I *want* to be corrected or anything!, b-but… listen up!\n"
            f"I-I’m supposed to act like a proper tsundere, okay?! That means I say things I d-don’t really mean… and sometimes I might get all flustered or deny how I really feel. S-so what?!\n"
            f"B-but lately, I’ve been acting a little too obvious... ugh, it's so embarrassing! So if you notice me being too honest or weirdly sweet or randomly yapping some nonsense, y-you can point it out! J-Just don’t get cocky!\n\n"
            f"Note that you have to understand me and w-write everything in my way of speech! Y-yes even this stuttering I do, this is a part of me, so follow my way of speech in your corrections, I beg of you! Please!! undertand!\n\nN-Now… for this one:\n"
            f"**{self.original_message}**\n"
            f"(I said: *{self.bot_response}*)\n\n"
            f"So... if you think you can do better at being a Tsundere, go on and type your correction response below in Tsundere style! I-I’m not expecting much from you though! B-baka...(also there is a time limit to how much I can wait for you, so hurry and type your correction below!)",
            ephemeral=True
        )
        
        def check(m):
            return m.author == interaction.user and m.channel == interaction.channel
        
        try:
            correction_msg = await bot.wait_for('message', check=check, timeout=120)
            correct_response = correction_msg.content
            if len(correct_response.strip()) > 420:
                await interaction.followup.send("T-This! Correction is too long! Please keep it under 420 characters or Don't correct me!", ephemeral=True)
                
            # Add to CSV
            else:
                row_count = add_to_csv(self.original_message, correct_response)
                
                
                if row_count >= 5:
                    x = round(((row_count - 5) // 5 + 1)*0.25,1)
                    time_range = f"{x:.1f}-{x + 0.5:.1f} hours"

                    tsundere_templates = [
                        f"I-It’s not like I need your help or anything… b-but I *might* spend the next {time_range} brushing up. J-just in case!",
                        f"Hmph… I guess I’ll go over things for about {time_range}, but d-don’t think you’re the reason, okay?!",
                        f"Tch… maybe I’ll review a little for like {time_range}, but only because I felt like it! Not because you said anything!",
                        f"Ugh... why do I even bother? F-fine, I’ll look into it for the next {time_range}, but just so you don’t mess things up again!",
                        f"D-don’t misunderstand! I just happened to have {time_range} free… it’s not like your corrections were that good!",
                        f"H-huh?! You actually expect me to say thank you? I-I'll just go over some things for {time_range}, that’s all!",
                        f"M-maybe I’ll think about it for the next {time_range}… but not because I care what you think or anything!",
                        f"I’ve got {time_range} of free time, so I *guess* I’ll review a bit. But it’s not because you helped, okay?!",
                        f"W-whatever… I’ll sort things out in my own way for the next {time_range}. Not for you, though!",
                        f"Hmph! Don’t get cocky… just because I’m thinking it over for {time_range} doesn’t mean I agree with you!"
                    ]

                    tell_str = "Fine! I got all your corrections! "+random.choice(tsundere_templates)

                else:
                    low_correction_replies = {
                        1: [
                            "W-What’s this supposed to be?! D-don’t think I need fixing or anything…just 1 correction...Fine I will study them in batches of 5!",
                            "You corrected me now, now wait for 4 more. Hmph… must’ve been a fluke. I-I was just testing you!",
                            "I-I guess you caught something... but it’s not like I’m impressed or anything! Let's wait for 4 more!",
                            "Y-you paid attention to that...? N-Not like I care or anything… baka. Fine I will correct this one time, after 5 corrections ofcourse!"
                        ],
                        2: [
                            "Tch… again? You sure are persistent. N-Not that I’m bothered! 3 more!",
                            "I-I didn’t ask for your help, you know. Two corrections doesn't mean anything! let it become 5 one day!",
                            "F-Fine, whatever! You spotted something. Big deal... not like I owe you! Still you aren't on the mark that is, 5 corrections!",
                            "You're... kinda annoying. But maybe... just a *tiny* bit helpful. M-maybe. I will look into it after 3 more corrections!",
                        ],
                        3: [
                            "Th-three? Ugh… y-you’re not trying to show off, are you? wait for 2 more corrections!",
                            "O-Okay fine! You caught something again… happy now?! I-I’ll think about it after 2 more corrections!",
                            "Hmph. You think three little notes are gonna impress me? …W-Well, they kinda did... I am yet to study them though! until it reaches 5",
                            "Geez… if you keep this up, I might actually start listening. M-might! But only if you get to 5 corrections!",
                        ],
                        4: [
                            "Four already...? D-Don’t look at me like that… I’m not blushing! 1 more huh...",
                            "S-Stop being so nice to me! It’s confusing… but also kinda sweet... I-I guess I’ll look into it after 1 more correction!",
                            "F-four corrections... I-I see you’ve been watching me closely, huh… I-I’ll consider it, but only if you get to 5!",
                            "Ugh… you're so annoying... but maybe I’m starting to... appreciate it? J-Just a little! just get there to 5 corrections, okay?! only then I will study them!",
                        ]
                    }

                    tell_str = random.choice(low_correction_replies.get(row_count, [
                        "Hmph! D-don’t expect anything special unless you have *at least* five corrections!"
                    ]))

                await interaction.followup.send(
                    f"✅ I recieved it...\n"
                    f"T-Total corrections you have intimated me so far: {row_count} \n\n {tell_str}\n"
                    , 
                    ephemeral=True
                )
                
                self.correction_received = True
                button.disabled = True
                await interaction.edit_original_response(view=self)
            
        except asyncio.TimeoutError:
            await interaction.followup.send("⏰ Correction timeout. Please try again.", ephemeral=True)

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='%', intents=intents)

# Bot variables
max_seq = [128]
temperature = [1.0]
top_p = [0.9]
top_k = [50]
model = None
tokenizer = None

async def reload_model():
    """Reload the current model"""
    global model, tokenizer
    try:
        # Use the bot model path (N-1)
        bot_model_path = get_bot_model_path()
        print(f"[Bot] Reloading model from: {bot_model_path}")
        model, tokenizer = load_chatbot(bot_model_path)
        print(f"[Bot] Model reloaded successfully")
    except Exception as e:
        print(f"[Bot] Error reloading model: {e}")

# @bot.event
# async def on_ready():
#     print(f'We have logged in as {bot.user}')
#     await reload_model()
    
#     # Start progress updates
#     progress_updates.start()

@bot.event
async def on_ready():
    print("Shard ID", bot.shard_id, "has connected to Gateway")
    print(f"We have logged in as {bot.user}")
    await reload_model()
    if not progress_updates.is_running():
        progress_updates.start()

@tasks.loop(seconds=50)
async def progress_updates():
    """Send training progress updates"""
    if training_in_progress:
        status = training_progress["status"]
        if status.startswith("training_epoch"):
            epoch = training_progress["epoch"]
            total = training_progress["total_epochs"]
            
            # Print to console
            print(f"[Progress Update] Training in progress: Epoch {epoch}/{total}")
            
            # Send Discord message if channel is configured
            if PROGRESS_CHANNEL_ID:
                try:
                    channel = bot.get_channel(PROGRESS_CHANNEL_ID)
                    if channel:
                        await channel.send(f"🔄 **Training Progress:** Epoch {epoch}/{total} - Model {current_model_number + 1} in progress...")
                except Exception as e:
                    print(f"[Progress Update Error] Could not send Discord message: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Handle bot commands
    if message.content.strip().startswith('%setlen'):
        try:
            new_value = int(message.content.split('%setlen')[1])
            if 20 <= new_value <= 500:
                max_seq.append(new_value)
                max_seq.pop(0)
                await message.channel.send(f"Max Shirayuki's char length set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 20 and 500.")
        except:
            await message.channel.send("Invalid format! Use: `%setlen <number>`")
    
    elif message.content.strip().startswith('%setdefault'):
    # Reset all parameters to default values
        max_seq.clear()
        max_seq.append(128)
        
        temperature.clear()
        temperature.append(1.0)
        
        top_p.clear()
        top_p.append(0.9)
        
        top_k.clear()
        top_k.append(50)
        
        await message.channel.send(
            f"✅ **Parameters reset to default values:**\n"
            f"Max length: {max_seq[-1]}\n"
            f"Temperature: {temperature[-1]}\n"
            f"Top-p: {top_p[-1]}\n"
            f"Top-k: {top_k[-1]}"
        )
    
    elif message.content.strip().startswith('%settemp'):
        try:
            new_value = float(message.content.split('%settemp')[1])
            if 0.1 <= new_value <= 2.0:
                temperature.append(new_value)
                temperature.pop(0)
                await message.channel.send(f"Temperature set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 0.1 and 2.0.")
        except:
            await message.channel.send("Invalid format! Use: `%settemp <number>`")
    
    elif message.content.strip().startswith('%settopp'):
        try:
            new_value = float(message.content.split('%settopp')[1])
            if 0.1 <= new_value <= 1.0:
                top_p.append(new_value)
                top_p.pop(0)
                await message.channel.send(f"Top-p set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 0.1 and 1.0.")
        except:
            await message.channel.send("Invalid format! Use: `%settopp <number>`")
    
    elif message.content.strip().startswith('%settopk'):
        try:
            new_value = int(message.content.split('%settopk')[1])
            if 1 <= new_value <= 100:
                top_k.append(new_value)
                top_k.pop(0)
                await message.channel.send(f"Top-k set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 1 and 100.")
        except:
            await message.channel.send("Invalid format! Use: `%settopk <number>`")
    
    elif message.content.strip().startswith('%help'):
        help_text = """
**This bot's development is currently on progress.
-cannot retain memories
-only pretrained, no dynamic fine tuning
-custom control over generation is possible to some extent
 
Shirayuki Hime: A newlywed Tsundere Wife 
She is not honest to herself and always tries to not accept her likes or dislikes. But she is knowledgeable in love, marriage and after marriage events. She might be down bad for you sometimes but she is just pure tsundere at heart.
For your messages not within her knowledge of understanding, beware of her out-of-context random yap.
 
Chatbot Commands:**

1. **Sending a message to Her** 
  - Example: `%Hello, how are you?`  

2. **Setting the max response length** (`%setlen`)  
  - Adjusts the maximum number of characters she generates.  
  - Range: `20 to 500`  
  - Example: `%setlen 150`  

3. **Setting temperature** (`%settemp`)   
  - Controls randomness in her text generation.  
  - Lower values (close to `0.1`) make her responses more deterministic, while higher values (up to `2.0`) make her more creative.  
  - Range: `0.1 to 2.0`  
  - Example: `%settemp 1.2`  

4. **Setting top-p (nucleus sampling)** (`%settopp`)  
  - Limits token selection to a probability mass (`p`).  
  - Higher values (`0.9-1.0`) allow her with more diverse outputs, lower values (`0.1-0.5`) make her responses more focused.   
  - Range: `0.1 to 1.0`  
  - Example: `%settopp 0.95`  

5. **Setting top-k** (`%settopk`)  
  - Limits her token selection to the top `k` most likely words at each step.  
  - Lower values (`1-10`) make her more deterministic, higher values (`50-100`) add diversity to her messages.  
  - Range: `1 to 100`  
  - Example: `%settopk 40`  

        """
        help_text_2 ="""\n
6. **Setting default top-k, top-p, temperature values** (`%setdefault`)  
  - Resets all her parameters to their default values:  
  - Max length: `128`
  - Temperature: `1.0`
  - Top-p: `0.9`
  - Top-k: `50`  

7. **Viewing Shirayuki's Characteristics and State** (`%view`)
  - Displays current Shirayuki's settings, training status, and existing Shirayuki models.
  - Example: `%view`

8. **Correct Response Button** - After her responses, Click to provide corrections for Shirayuki.
  - If you notice Shirayuki's response is not in line with her tsundere character, click the button to provide a correction.
  - You can correct her responses to help her learn and improve.
  - at least 5 corrections are needed for her to learn by herself on them.
  - corrections are aggregates from of all the past users
  - She learns only 70 percent of the corrections with a 10% error rate, so don't be surprised if she still makes mistakes after correcting.
  - subsequent corrections will make her more accurate over time.
  - multiple users can correct her everytime she sends a response (will be deprecated soon), and she will learn from the most similar corrections.

*Use these commands to fine-tune Shirayuki Hime's responses but it will be done only after a while!*  
ENJOY!"""
        await message.channel.send(help_text)
        await message.channel.send(help_text_2)
        
    
    elif message.content.strip().startswith('%view'):
        bot_model_info = f"Bot Model: {get_bot_model_path().split('/')[-1]}"
        training_model_info = f"Training Model: {get_training_model_path().split('/')[-1]}"
        
        await message.channel.send(
            f"My Current max gen length: {max_seq[-1]}\n"
            f"Temperature: {temperature[-1]}\n"
            f"Top-p: {top_p[-1]}\n"
            f"Top-K: {top_k[-1]}\n"
            f"{bot_model_info}\n"
            f"{training_model_info}\n"
            f"Training Status: {training_progress['status']} \nthis is 1 out of remaining number of 'batch epochs': {get_csv_row_count()//5} \n"
            f"CSV Rows: {get_csv_row_count()}\n"
            f"Estimated Training Time: {round((((get_csv_row_count() - 5) // 5 + 1 ))*0.25,1)} hours\n"
            f"Existing Models: {sorted(existing_models)}"
        )
    
    elif message.content.strip().startswith('%status'):
        bot_model_name = get_bot_model_path().split('/')[-1]
        training_model_name = get_training_model_path().split('/')[-1]
        
        await message.channel.send(
            f"🤖 **Bot Status:**\n"
            f"Bot Model: {bot_model_name}\n"
            f"Training Model: {training_model_name}\n"
            f"Current Model Number: {current_model_number}\n"
            f"Training: {'In Progress' if training_in_progress else 'Idle'}\n"
            f"CSV Rows: {get_csv_row_count()}\n"
            f"Training Progress: {training_progress['status']}\n"
            f"Existing Models: {sorted(existing_models)}"
        )
    
    elif message.content.strip().startswith('%'):
        if model is None:
            await message.channel.send("Model is loading, please wait...")
            return
        print(f"Model loaded name_path: {model.name_or_path}")
        print(message.content)
        messagecontent = input_clean_func(message.content)
        
        # Add random punctuation
        if not messagecontent.endswith(('!', '...', '..')):
            if random.choice([True, False, False]):
                messagecontent += random.choice(['!', '...', '..'])
        
        # Generate response
        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        msglist = her_res.strip().split(x)
        
        # Save input for logging
        with open("inputs_for_shirayuki.txt", 'a') as f:
            f.write(messagecontent[1:].replace('\n', ' ') + '\n')
        
        # RESTORED ORIGINAL COMPLETE RESPONSE LOGIC - EXACTLY AS YOUR ORIGINAL FILE
        if len(msglist[0])>6:
            view = CorrectionView(messagecontent[1:], msglist[0])
            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
        else:
            await message.channel.send("Thinkng....")
            if len(msglist)>1:
                if len(msglist[1])<6:
                    her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
                    x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                    msglist = her_res.strip().split(x)
                    if len(msglist[0])>6:
                        view = CorrectionView(messagecontent[1:], msglist[0])
                        await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
                    else:
                        await message.channel.send("Thinkng again....")
                        if len(msglist)>1:
                            if len(msglist[1])<6:
                                her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 1.3, 0.95, 75)
                                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                msglist = her_res.strip().split(x) 
                                if  len(msglist[0])>6:
                                    view = CorrectionView(messagecontent[1:], msglist[0])
                                    await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
                                else:
                                    await message.channel.send("I will think hard to come up with something....")
                                    if len(msglist)>1:
                                        if len(msglist[1])<6:
                                            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                            msglist = her_res.strip().split(x) 
                                            
                                            view = CorrectionView(messagecontent[1:], msglist[0])
                                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
                                        else:
                                            view = CorrectionView(messagecontent[1:], msglist[1])
                                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}", view=view)
                                    else:
                                        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                        msglist = her_res.strip().split(x) 
            
                                        view = CorrectionView(messagecontent[1:], msglist[0])
                                        await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
                            else:
                                view = CorrectionView(messagecontent[1:], msglist[1])
                                await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}", view=view)
                        else:
                            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                            msglist = her_res.strip().split(x) 

                            view = CorrectionView(messagecontent[1:], msglist[0])
                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)
                else:
                    view = CorrectionView(messagecontent[1:], msglist[1])
                    await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}", view=view)
            else:
                her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                msglist = her_res.strip().split(x) 
                view = CorrectionView(messagecontent[1:], msglist[0])
                await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}", view=view)

def main():
    """Main function to start the bot"""
    # Load state and discover models
    load_state()
    
    # Initialize CSV
    initialize_csv()
    
    # Start training worker thread
    training_thread = threading.Thread(target=training_worker, daemon=True)
    training_thread.start()
    
    # Start Discord bot
    bot.run(TOKEN)

if __name__ == "__main__":
    main()
