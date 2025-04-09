token = 'MTM1NDgxNDgxODQ1NTkxMjU1OQ.GPDkcb.KWgJx_1FB81Xxx3ItUL1U2AGBEXrNieUx5e5zs'

# import discord
# #import os
# import requests
# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# def load_chatbot(model_path='./fine_tuned_ShirayukiOPT350' ): #"./fine_tuned_AishaGPT2"):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForCausalLM.from_pretrained(model_path)
#     return model, tokenizer

# # def chat(model, tokenizer):
# #     print("Chatbot is ready! Type 'exit' to quit.")
# #     while True:
# #         user_input = input("You: ")
# #         if user_input.lower() == "exit":
# #             break
        
# #         input_text = f"<|startoftext|>{user_input}<|sep|>"
# #         input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
# #         with torch.no_grad():
# #             output_ids = model.generate(input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)
        
# #         response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# #         response = response.split("<|sep|>")[-1]  # Extract the chatbot's reply
# #         print(f"Aisha Nekron: {response.strip()}")
        
# def get_Aisha_response(model, tokenizer, user_input, max_seq=128):
#     input_text = f"<|startoftext|>{user_input}<|sep|>"
#     input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
#     with torch.no_grad():
#         output_ids = model.generate(input_ids, max_length=max_seq, pad_token_id=tokenizer.eos_token_id)
    
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     response = response.split("<|sep|>")[-1]  #Extract the chatbot's reply
#     return response.strip()

# # def get_quote():
# #   response = requests.get("https://zenquotes.io/api/random")
# #   json_data = json.loads(response.text)
# #   quote = json_data[0]['q'] + " -" + json_data[0]['a']
# #   return(quote)

# max_seq=[128]
# model, tokenizer = load_chatbot()

# intents = discord.Intents.default()
# intents.message_content = True
# client = discord.Client(intents=intents)

# @client.event
# async def on_ready():
#   print('We have logged in as {0.user}'.format(client))


# @client.event
# async def on_message(message):
#   if message.author == client.user:
#     return

#   if message.content.startswith('$set'):
#     if 20 <= int(message.content.split('$set')[1]) <=500:
#       max_seq.append(int(message.content.split('$set')[1]))
#       await message.channel.send("max Aisha's chr length set to:"+str(max_seq[-1]))
#       try:
#         max_seq.pop(0)
#       except:
#         print("error popping")
#     else:
#       await message.channel.send("out of range chr length has been set")
      
    
    
    

#   elif message.content.startswith('$'):
    
#     print(message.content)
#     her_res= get_Aisha_response(model,tokenizer, message.content,max_seq[-1])
#     await message.channel.send(#"<response to> "+ message.content[1:]+
#                                #"\n\nAisha Nekron: "+
#                                "%"+her_res)
  
  
    

import discord
import requests
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# with open("inputs_for_shirayuki.txt", 'a') as f:
#     print("question file created/connected to!")

def load_chatbot(model_path="./fine_tuned_ShirayukiOPT350"):
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

    if message.content.strip().startswith('$setlen'):
        try:
            new_value = int(message.content.split('%$etlen')[1])
            if 20 <= new_value <= 500:
                max_seq.append(new_value)
                max_seq.pop(0)
                await message.channel.send(f"Max Aisha's char length set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 20 and 500.")
        except:
            await message.channel.send("Invalid format! Use: `$setlen <number>`")

    elif message.content.strip().startswith('$settemp'):
        try:
            new_value = float(message.content.split('$settemp')[1])
            if 0.1 <= new_value <= 2.0:
                temperature.append(new_value)
                temperature.pop(0)
                await message.channel.send(f"Temperature set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 0.1 and 2.0.")
        except:
            await message.channel.send("Invalid format! Use: `$settemp <number>`")

    elif message.content.strip().startswith('$settopp'):
        try:
            new_value = float(message.content.split('$settopp')[1])
            if 0.1 <= new_value <= 1.0:
                top_p.append(new_value)
                top_p.pop(0)
                await message.channel.send(f"Top-p set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 0.1 and 1.0.")
        except:
            await message.channel.send("Invalid format! Use: `$settopp <number>`")

    elif message.content.strip().startswith('$settopk'):
        try:
            new_value = int(message.content.split('$settopk')[1])
            if 1 <= new_value <= 100:
                top_k.append(new_value)
                top_k.pop(0)
                await message.channel.send(f"Top-k set to: {new_value}")
            else:
                await message.channel.send("Out of range! Set a value between 1 and 100.")
        except:
            await message.channel.send("Invalid format! Use: `$settopk <number>`")
    
    elif message.content.strip().startswith('$help'):
        help_text = """
    **Aisha Nekron Chatbot Commands:**

    1. **Sending a message to Her**  
    - Simply type `$<your message>`  
    - Example: `$Hello, how are you?`  

    2. **Setting the max response length** (`$setlen`)  
    - Adjusts the maximum number of characters she generates.  
    - Usage: `$setlen <number>`  
    - Range: `20 to 500`  
    - Example: `$setlen 150`  

    3. **Setting temperature** (`$settemp`)  
    - Controls randomness in generation.  
    - Lower values (close to `0.1`) make responses more deterministic, while higher values (up to `2.0`) make them more creative.  
    - Usage: `$settemp <value>`  
    - Range: `0.1 to 2.0`  
    - Example: `$settemp 1.2`  

    4. **Setting top-p (nucleus sampling)** (`$settopp`)  
    - Limits token selection to a probability mass (`p`).  
    - Higher values (`0.9-1.0`) allow more diverse outputs, lower values (`0.1-0.5`) make responses more focused.  
    - Usage: `$settopp <value>`  
    - Range: `0.1 to 1.0`  
    - Example: `$settopp 0.95`  

    5. **Setting top-k** (`$settopk`)  
    - Limits token selection to the top `k` most likely words at each step.  
    - Lower values (`1-10`) make it more deterministic, higher values (`50-100`) add diversity.  
    - Usage: `$settopk <number>`  
    - Range: `1 to 100`  
    - Example: `$settopk 40`  
    
    6. **Viewing Aisha's Characteristics** (`$view`)
    -Usage: $view
    -Example: My Current max gen length:<>
     Temperature:<>
     Top-p:<>
     Top-K:<>
     

    *Use these commands to fine-tune Aisha Nekron's responses!*  
        """
        await message.channel.send(help_text)
    
    elif message.content.strip().startswith('$view'):
        await message.channel.send("My Current max gen length:"+str(max_seq[-1])+"\nTemperature:"+str(temperature[-1])+"\nTop-p:"+str(top_p[-1])+"\nTop-K:"+str(top_k[-1]))
        
        
    elif message.content.strip().startswith('$'):
        print(message.content)
        time.sleep(5)
        her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
        msglist = her_res.strip().split(x)

        # with open("inputs_for_shirayuki.txt", 'a') as f:
        #     f.write(message.content[1:].replace('\n', ' ') + '\n')
        if len(msglist[0])>6:
            #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
            await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
        else:
            #await message.channel.send("Thinkng....")
            if len(msglist)>1:
                if len(msglist[1])<6:
                    her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
                    x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                    msglist = her_res.strip().split(x)
                    if len(msglist[0])>6:
                        #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                        await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
                    else:
                        #await message.channel.send("Thinkng again....")
                        if len(msglist)>1:
                            if len(msglist[1])<6:
                                her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 1.3, 0.95, 75)
                                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                msglist = her_res.strip().split(x) 
                                if  len(msglist[0])>6:
                                    #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                                    await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
                                else:
                                    #await message.channel.send("I will think hard to come up with something....")
                                    if len(msglist)>1:
                                        if len(msglist[1])<6:
                                            her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
                                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                            msglist = her_res.strip().split(x) 
                                            
                                            #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                                            await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
                                        else:
                                            #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
                                            await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[1]}")
                                    else:
                                        her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
                                        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                        msglist = her_res.strip().split(x) 
            
                                        #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                                        await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
                            else:
                                #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
                                await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[1]}")
                        else:
                            her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                            msglist = her_res.strip().split(x) 

                            #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                            await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[1]}")
                else:
                    #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[1]}")
                    await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[1]}")
            else:
                #await message.channel.send("I will think hard to come up with something....")
                her_res = get_Shirayuki_response(model, tokenizer, message.content, max_seq[-1], 2.0, 0.99, 99)
                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                msglist = her_res.strip().split(x) 
                #await message.channel.send(f"\n<response to> {message.content[1:]}\n\nShirayuki Hime: {msglist[0]}")
                await message.channel.send(#f"\n<response to> {message.content[1:]}\n\nShirayuki Hime:
                                       f"%{msglist[0]}")
    
                    
            

client.run(token)