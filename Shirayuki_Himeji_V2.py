import discord
import requests
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import time

def input_clean_func(input_string):
    
    # Regular expression to match Discord mentions format: <@numbers>
    mention_pattern = r'<@\d+>'
    # Replace all occurrences of the pattern with "them"
    cleaned_input = re.sub(mention_pattern, "them", input_string)
    
    return cleaned_input

def input_clean_func(input_string):
    
    # Regular expression to match Discord mentions format: <@numbers>
    mention_pattern = r'<@\d+>'
    # Replace all occurrences of the pattern with "them"
    cleaned_input = re.sub(mention_pattern, "them", input_string)
    
    return cleaned_input

# Uncomment the following lines to create or connect to a file for storing inputs
# with open("inputs_for_shirayuki.txt", 'a') as f:
#     print("question file created/connected to!")

def load_chatbot(model_path="./fine_tuned_ShirayukiV4_OPT"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def get_Shirayuki_response(model, tokenizer, user_input, max_seq=128, temp=1.0, top_p=0.9, top_k=50):
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

# Default settings
max_seq = [128]
temperature = [1.0]
top_p = [0.9]
top_k = [50]

model, tokenizer = load_chatbot()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

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
    - Controls randomness in generation.  
    - Lower values (close to `0.1`) make responses more deterministic, while higher values (up to `2.0`) make them more creative.  
    - Range: `0.1 to 2.0`  
    - Example: `%settemp 1.2`  
 3
    4. **Setting top-p (nucleus sampling)** (`%settopp`)  
    - Limits token selection to a probability mass (`p`).  
    - Higher values (`0.9-1.0`) allow more diverse outputs, lower values (`0.1-0.5`) make responses more focused.  
    - Range: `0.1 to 1.0`  
    - Example: `%settopp 0.95`  

    5. **Setting top-k** (`%settopk`)  
    - Limits token selection to the top `k` most likely words at each step.  
    - Lower values (`1-10`) make it more deterministic, higher values (`50-100`) add diversity.  
    - Range: `1 to 100`  
    - Example: `%settopk 40`  
    
    6. **Viewing Shirayuki's Characteristics** (`%view`)
    *Use these commands to fine-tune Shirayuki Hime's responses!*  
    ENJOY!
        """
        await message.channel.send(help_text)
    
    elif message.content.strip().startswith('%view'):
        await message.channel.send("My Current max gen length:"+str(max_seq[-1])+"\nTemperature:"+str(temperature[-1])+"\nTop-p:"+str(top_p[-1])+"\nTop-K:"+str(top_k[-1]))
        
    """all conditions below are for Shirayuki Hime's response generation"""
    elif message.content.strip().startswith('%'):
        print(message.content)
        messagecontent=input_clean_func(message.content)
        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        msglist = her_res.strip().split(x)

        # Uncomment the following lines to create or connect to a file for storing inputs
        with open("inputs_for_shirayuki.txt", 'a') as f:
            f.write(messagecontent[1:].replace('\n', ' ') + '\n')
            
            
        if len(msglist[0])>6:
            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
        else:
            await message.channel.send("Thinkng....")
            if len(msglist)>1:
                if len(msglist[1])<6:
                    her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
                    x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                    msglist = her_res.strip().split(x)
                    if len(msglist[0])>6:
                        await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
                    else:
                        await message.channel.send("Thinkng again....")
                        if len(msglist)>1:
                            if len(msglist[1])<6:
                                her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 1.3, 0.95, 75)
                                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                msglist = her_res.strip().split(x) 
                                if  len(msglist[0])>6:
                                    await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
                                else:
                                    await message.channel.send("I will think hard to come up with something....")
                                    if len(msglist)>1:
                                        if len(msglist[1])<6:
                                            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                            msglist = her_res.strip().split(x) 
                                            
                                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
                                        else:
                                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}")
                                    else:
                                        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                        msglist = her_res.strip().split(x) 
            
                                        await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
                            else:
                                await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}")
                        else:
                            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                            msglist = her_res.strip().split(x) 

                            await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
                else:
                    await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[1]}")
            else:
                her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                msglist = her_res.strip().split(x) 
                await message.channel.send(f"\n<response to> {messagecontent[1:]}\n\nShirayuki Hime: {msglist[0]}")
        
        """Normal mode of operation to interact wuth Aisha bot, uncomment to use"""
        # print(message.content)
        # time.sleep(5)
        # her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
        # x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        # msglist = her_res.strip().split(x)

        # # with open("inputs_for_shirayuki.txt", 'a') as f:
        # #     f.write(message.content[1:].replace('\n', ' ') + '\n')
        # if len(msglist[0])>6:
        #     #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #     await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
        # else:
        #     #await message.channel.send("Thinkng....")
        #     if len(msglist)>1:
        #         if len(msglist[1])<6:
        #             her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
        #             x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #             msglist = her_res.strip().split(x)
        #             if len(msglist[0])>6:
        #                 #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #                 await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
        #             else:
        #                 #await message.channel.send("Thinkng again....")
        #                 if len(msglist)>1:
        #                     if len(msglist[1])<6:
        #                         her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 1.3, 0.95, 75)
        #                         x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #                         msglist = her_res.strip().split(x) 
        #                         if  len(msglist[0])>6:
        #                             #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #                             await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
        #                         else:
        #                             #await message.channel.send("I will think hard to come up with something....")
        #                             if len(msglist)>1:
        #                                 if len(msglist[1])<6:
        #                                     her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
        #                                     x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #                                     msglist = her_res.strip().split(x) 
                                            
        #                                     #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #                                     await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
        #                                 else:
        #                                     #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
        #                                     await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[1]}")
        #                             else:
        #                                 her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
        #                                 x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #                                 msglist = her_res.strip().split(x) 
            
        #                                 #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #                                 await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
        #                     else:
        #                         #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
        #                         await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[1]}")
        #                 else:
        #                     her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
        #                     x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #                     msglist = her_res.strip().split(x) 

        #                     #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #                     await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[1]}")
        #         else:
        #             #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
        #             await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[1]}")
        #     else:
        #         #await message.channel.send("I will think hard to come up with something....")
        #         her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
        #         x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        #         msglist = her_res.strip().split(x) 
        #         #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
        #         await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
        #                                f"${msglist[0]}")
    
                    
            

token = "YOUR_DISCORD_BOT_TOKEN"  # Replace with your actual Discord bot token
client.run(token)
