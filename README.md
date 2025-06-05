# Custom_Personality_Bot
A tested personality for this chatbot is a tsundere, a character archetype that flips between cold and warm behavior, always acts opposite to internal feelings, does unhonest denials and gets embarrassed easily. This project enables the training and deployment of custom personality-based chatbots using conversation datasets and fine-tuned language models.


____________________________________________________________________________________________________________________________________________________________________________________________________________________



💡 Overview
This repo provides a framework for developing character-driven chatbots powered by fine-tuned models. Whether you're building a flirtatious anime companion or a formal assistant, you can create a bot that speaks in a specific tone, personality, or communication style using structured dialogue datasets.

Models used:
https://huggingface.co/facebook/opt-350m
https://huggingface.co/openai-community/gpt2

____________________________________________________________________________________________________________________________________________________________________________________________________________________






📁 Project Structure
Below are the key files included in this repository:

tsundere.py: Contains a list of dialogue tuples formatted as (user, bot response).

augment_csv_data.py: Augments existing datasets to increase variety and improve model robustness.

augmented_conversation_dataset_ShirayukiV2.csv: Augmented dataset generated from the base conversation.

chat_bot.py: Basic chatbot runner for interacting with a trained model.

conversation_dataset.csv: Original conversation dataset used as a base.

conversation_dataset_Shirayuki.csv: Refined version of the dataset with specific personality traits.

conversation_dataset_ShirayukiV2.csv: Further refined dataset used for final fine-tuning.

convert_to_csv.py: Utility for converting raw tuple data into CSV format.

custom_chatbot.py: Script for deploying a personality-driven chatbot from trained models.

dynamic_chatbot.py: Enhanced chatbot runner with adjustable parameters.

fine_tune_gpt2.py: Script to fine-tune GPT-2 model on the custom dataset.

fine_tune_opt350m.py: Script to fine-tune Meta's OPT-350M model.

google_translator_transform.py: Adds multilingual capability to the dataset using Google Translate.

inputs_for_shirayuki.txt: Sample input prompts for testing.

recover_checkpoint.py: Used for recovering and continuing training from checkpoints.

Shirayuki_Himeji_V2.py: Example chatbot implementation using a specific trained model.

Aisha_NekronV2.py: Another character implementation with its own trained model.

download_lang_model.py: Utility for downloading tokenizer and language models.

whatsapp_bot.py: A demonstration to release the bot in whatsapp.



____________________________________________________________________________________________________________________________________________________________________________________________________________________




🚀 How to Use
Prepare Data
Use tsundere.py or your own dataset. Convert it to CSV using convert_to_csv.py.

Augment Dataset (optional)
Run augment_csv_data.py to expand the dataset with slight variations.

Train a Model
Choose a training script:

fine_tune_gpt2.py for GPT-2

fine_tune_opt350m.py for OPT-350M

Test Your Bot
Use chat_bot.py or dynamic_chatbot.py to chat with the bot interactively.

Deploy a Personality
Create a deployment script like Shirayuki_Himeji_V2.py to load your fine-tuned model and start a custom session.




____________________________________________________________________________________________________________________________________________________________________________________________________________________





🔧 Requirements
Python 3.8+

PyTorch

Transformers (HuggingFace)

pandas, numpy

tqdm

Google Translate API (optional)




____________________________________________________________________________________________________________________________________________________________________________________________________________________




Install dependencies:

in bash:
pip install -r requirements.txt

____________________________________________________________________________________________________________________________________________________________________________________________________________________



📌 Notes
Keep your dataset consistent with the (user, bot_response) format.

Dataset diversity and quantity directly influence response quality.

You can create multiple characters by modifying datasets and personality-driven vocabulary.
