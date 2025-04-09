import pandas as pd
import random
from deep_translator import GoogleTranslator
import time
from tqdm import tqdm
from multiprocessing import Pool
import copy
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#dic={'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'}

# def back_translate(text, src="en", dest="en"):
#     """Translate text to another language and back to introduce variation."""
#     mid = random.choice(list(dic.values()))
#     translated = GoogleTranslator(source=src, target=mid).translate(text)
#     back_translated = GoogleTranslator(source=mid, target=dest).translate(translated)
#     return back_translated


# def back_translate(text, src="en", dest="en"):
#     """Translate text to another language and back to introduce variation."""
#     for _ in range(3):  # Try up to 3 times with different languages
#         try:
#             mid = random.choice(list(dic.values()))
#             time.sleep(0.15)
#             translated = GoogleTranslator(source=src, target=mid).translate(text)
#             time.sleep(0.25)
#             back_translated = GoogleTranslator(source=mid, target=dest).translate(translated)
#             return back_translated
#         except Exception as e:
#             print(f"Translation failed: {e}")
#             time.sleep(1)  # Prevent API overload
#     return text  # Return original text if translation fails



# Load model and tokenizer once (global variables for reuse)
_model = None
_tokenizer = None

def load_model_if_needed():
    """Load the model and tokenizer if they haven't been loaded yet."""
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        model_name = "facebook/m2m100_418M"
        cache_dir = "./model_cache"
        
        _tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=True)
        _model = M2M100ForConditionalGeneration.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            use_safetensors=False,local_files_only=True
        )
    
    return _model, _tokenizer

def translate_text(text, source_lang, target_lang):
    """Translate text from source language to target language using M2M100."""
    model, tokenizer = load_model_if_needed()
    
    # Set source language
    tokenizer.src_lang = source_lang
    
    # Encode the text
    encoded = tokenizer(text, return_tensors="pt")
    
    # Generate translation with the target language
    tokenizer.tgt_lang = target_lang
    generated_tokens = model.generate(
        **encoded, 
        forced_bos_token_id=tokenizer.get_lang_id(target_lang),
        max_length=512  # Adjust as needed for longer texts
    )
    
    # Decode the generated tokens
    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

def back_translate(text, src="en", dest="en"):
    """Translate text to another language and back to introduce variation using M2M100."""
    # Supported languages in M2M100
    language_options = ["fr", "es", "de", "ru", "zh", "ar", "it", "nl", "pt", "ja"]
    
    for _ in range(5):  # Try up to 5 times with different languages
        try:
            # Choose a random intermediate language
            mid = random.choice(language_options)
            
            # Translate to intermediate language
            translated = translate_text(text, src, mid)
            
            # Translate back to original language
            back_translated = translate_text(translated, mid, dest)
            # if random.random() >0.9:
            #     print("Translated sentence:",translated)
            #     print("back translated:",back_translated)
            if text != back_translated:
                return back_translated
            else:
                mylist=copy.copy(language_options)
                mylist.remove(mid)
                mid = random.choice(mylist)
            
                # Translate to intermediate language
                translated = translate_text(text, src, mid)
                
                # Translate back to original language
                back_translated = translate_text(translated, mid, dest)
                return back_translated
            
        except Exception as e:
            print(f"Translation failed: {e}")
    print("failed 5 times!")
    return text  # Return original text if translation fails

def tsundere_variation(text):
    """Adds slight variations to a tsundere-style sentence."""
    stutter_chance = 0.35  # 35% chance of adding stuttering
    extra_words = ["B-baka!", "Ugh!", "H-hmph!", "Tch!", "I-I mean...", "D-don't misunderstand!","Well...", "Uh...", "You see...", "Listen...", "Like, seriously...", "Huh?", "Ehhh?",  
    "Wait a sec...", "So, um...", "Erm...", "H-hold on...", "Look...", "T-that's...",  
    "Sooo...", "A-anyway...", "J-just saying...", "I mean, like...", "Y-you know...",  
    "W-whatever...", "O-oh...","Umm...", "Eh?", "Uh-huh...", "O-oh yeah...", "Ah...", "Mmm...", "T-that is...",  
    "L-like I was saying...", "H-hang on...", "A-also...", "T-that reminds me...",  
    "S-sorta like...", "M-maybe...", "W-wait...", "B-but...", "Oh! Uh...", "A-anyhow...",  
    "S-something like that...", "Y-yeah, so...", "N-no, wait...", "O-or maybe not...",  
    "I-I guess...", "I-if you think about it...", "T-the thing is...", "A-ahem...",  
    "U-unless...", "W-well, never mind...", "A-about that...", "T-to be honest...",  
    "I-I dunno..."]
    
    end_words = [
    "...that's all.", "...you know.", "...or something.", "...I guess.", "...probably.",  
    "...maybe.", "...I dunno.", "...I suppose.", "...or whatever.", "...hmm.",  
    "...yeah.", "...so yeah.", "...anyway.", "...right?", "...I guess that works.",  
    "...makes sense, huh?", "...nothing special.", "...not a big deal.", "...no reason.",  
    "...just because.", "...or something like that.", "...it’s fine.", "...I don’t care.",  
    "...it doesn’t matter.", "...never mind.", "...if you say so.", "...I wasn’t asking.",  
    "...you wouldn’t get it.", "...whatever.", "...as if it matters.", "...just saying.",  
    "...well, whatever.", "...if that’s what you think.", "...not that it’s important.",  
    "...who cares?", "...don’t overthink it.", "...I guess that’s it.", "...so there.",  
    "...not like it means anything.", "...just leave it at that.", "...if that’s how it is.",  
    "...let’s not talk about it.", "...end of story.", "...so don’t ask.", "...that’s all I’ll say.",  
    "...and that’s that.", "...it’s nothing.", "...forget I said anything.", "...nothing more to add.",  
    "...nothing else to say."
    ]


    # words = text.split()
    # new_words = []
    
    # for word in words:
    #     if random.random() < stutter_chance and len(word) > 2:
    #         new_word = word[0] + "-" + word  # Stuttering effect
    #         new_words.append(new_word)
    #     else:
    #         new_words.append(word)

    # if random.random() > 0.6:
    #     new_words.insert(0, random.choice(extra_words))  # Add tsundere filler at the start
    # if random.random() > 0.6:
    #     new_words.append(random.choice(end_words))

    # return " ".join(new_words)
    
    


    words = text.split()
    new_words = []
    changed = False  # Flag to ensure at least one modification

    # Apply stuttering with an increasing chance if nothing has changed yet
    for word in words:
        if not changed and random.random() < stutter_chance + 0.2:  # Slightly higher chance initially
            if len(word) > 2:
                new_word = word[0] + "-" + word  # Stuttering effect
                new_words.append(new_word)
                changed = True
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    # Adjust probability for inserting tsundere filler at the start
    if changed:
        start_chance = 0.6  # Normal probability
    else:
        start_chance = 0.9  # Higher if nothing changed yet

    if random.random() < start_chance:
        new_words.insert(0, random.choice(extra_words))
        changed = True

    # Adjust probability for adding a tsundere end phrase
    if changed:
        end_chance = 0.6  # Normal probability
    else:
        end_chance = 0.95  # Very high if no change yet

    if random.random() < end_chance:
        new_words.append(random.choice(end_words))
        changed = True

    # Ensure at least one change happens (if all else fails, force one)
    if not changed:
        if len(new_words) > 1:
            new_words[1] = new_words[1][0] + "-" + new_words[1]  # Force stuttering on the second word
        else:
            new_words.append(random.choice(end_words))  # If it's a short sentence, add an end word

    return " ".join(new_words)

def process_row(row):
    """Process a single row for augmentation."""
    try:
        guy_sentence, girl_sentence = row["guy"], row["girl"]
        new_data = [(guy_sentence, girl_sentence)]
        
        for _ in range(2):
            paraphrased_guy = back_translate(guy_sentence)
            modified_girl = tsundere_variation(girl_sentence)
            new_data.append((paraphrased_guy, modified_girl))
        
        return new_data
    except Exception as e:
        print(f"Error processing row {row}: {e}")
        return []


# def augment_dataset(input_csv, output_csv, n=3):
#     """Augments the dataset by paraphrasing the guy's sentence and tsundere-modifying the girl's response."""
#     df = pd.read_csv(input_csv)
#     new_data = []
    
#     for _, row in df.iterrows():
#         if _ % 100==0:
#             print(_," rows processed...")
#         guy_sentence = row['guy']
#         girl_sentence = row['girl']
        
#         new_data.append((guy_sentence, girl_sentence))  # Keep original data
        
#         for _ in range(n):
#             paraphrased_guy = back_translate(guy_sentence)
#             modified_girl = tsundere_variation(girl_sentence)
#             new_data.append((paraphrased_guy, modified_girl))
    
#     # Convert to DataFrame and shuffle
#     new_df = pd.DataFrame(new_data, columns=['guy', 'girl'])
#     new_df = new_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
#     # Save to CSV
#     new_df.to_csv(output_csv, index=False)
#     print(f"Augmented dataset saved to {output_csv}")

# def augment_dataset(input_csv, output_csv, n=3):
#     """Augments dataset with paraphrasing and tsundere modifications."""
#     df = pd.read_csv(input_csv)
#     new_data = []
    
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
#         guy_sentence, girl_sentence = row['guy'], row['girl']
#         new_data.append((guy_sentence, girl_sentence))  # Keep original
        
#         for _ in range(n):
#             paraphrased_guy = back_translate(guy_sentence)
#             modified_girl = tsundere_variation(girl_sentence)
#             new_data.append((paraphrased_guy, modified_girl))
    
#     new_df = pd.DataFrame(new_data, columns=['guy', 'girl']).sample(frac=1).reset_index(drop=True)  # Shuffle
#     new_df.to_csv(output_csv, index=False)
#     print(f"Augmented dataset saved to {output_csv}")
def augment_dataset(input_csv, output_csv):
    """Augments dataset using multiprocessing."""
    df = pd.read_csv(input_csv)

    try:
        with Pool(processes=10 ) as pool:
            results = list(tqdm(pool.imap(process_row, df.to_dict(orient='records')), total=len(df), desc="Processing"))

        new_data = [item for sublist in results for item in sublist]
        new_df = pd.DataFrame(new_data, columns=['guy', 'girl']).sample(frac=1).reset_index(drop=True)
        new_df.to_csv(output_csv, index=False)
        print(f"Augmented dataset saved to {output_csv}")

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected! Closing pool...")
        pool.terminate()
        pool.join()
        exit(1)

# # Usage
input_csv = "conversation_dataset_Shirayuki.csv"  # Change to your input file
output_csv = "augmented_conversation_dataset_Shirayuki.csv"
if __name__ == "__main__":
    augment_dataset(input_csv, output_csv)
#augment_dataset(input_csv, output_csv)
