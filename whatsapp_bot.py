from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import discord
import requests
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# with open("inputs_for_shirayuki.txt", 'a') as f:
#     print("question file created/connected to!")

def load_chatbot(model_path="./fine_tuned_ShirayukiV3_OPT"):
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
# Initialize your bot model (replace with your own)
def get_bot_reply(user_msg):
    # Call your Shirayuki Hime model here instead
    her_res = get_Shirayuki_response(model, tokenizer, user_msg, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
    messagecontent=user_msg
    x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
    with open("inputs_for_shirayuki.txt", 'a') as f:
        f.write(messagecontent[0:].replace('\n', ' ') + '\n')
    msglist = her_res.strip().split(x)
    output=''
    if len(msglist[0])>6:
        output+=(f"\n  {msglist[0]}")
    else:
        output+="Thinkng...."
        if len(msglist)>1:
            if len(msglist[1])<6:
                her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], temperature[-1], top_p[-1], top_k[-1])
                x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                msglist = her_res.strip().split(x)
                if len(msglist[0])>6:
                    output+=(f"\n  {msglist[0]}")
                else:
                    output+="Thinkng again...."
                    if len(msglist)>1:
                        if len(msglist[1])<6:
                            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 1.3, 0.95, 75)
                            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                            msglist = her_res.strip().split(x) 
                            if  len(msglist[0])>6:
                                output+=(f"\n  {msglist[0]}")
                            else:
                                output+=("I will think hard to come up with something....")
                                if len(msglist)>1:
                                    if len(msglist[1])<6:
                                        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                        msglist = her_res.strip().split(x) 
                                        
                                        output+=(f"\n  {msglist[0]}")
                                    else:
                                        output+=(f"\n  {msglist[1]}")
                                else:
                                    her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                                    x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                                    msglist = her_res.strip().split(x) 
        
                                    output+=(f"\n  {msglist[0]}")
                        else:
                            output+=(f"\n  {msglist[1]}")
                    else:
                        her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
                        x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
                        msglist = her_res.strip().split(x) 

                        output+=(f"\n  {msglist[0]}")
            else:
                output+=(f"\n  {msglist[1]}")
        else:
            her_res = get_Shirayuki_response(model, tokenizer, messagecontent, max_seq[-1], 2.0, 0.99, 99)
            x = '<|endoftext|>' if '<|endoftext|>' in her_res.strip() else '<|endoftext|' if '<|endoftext|' in her_res.strip() else '<|endoftext' if '<|endoftext' in her_res.strip() else '<|endoftex' if '<|endoftex' in her_res.strip() else '<|endofte' if '<|endofte' in her_res.strip() else '<|endoft' if '<|endoft' in her_res.strip() else '<|endof' if '<|endof' in her_res.strip() else '<|endo' if '<|endo' in her_res.strip() else '<|end' if '<|end' in her_res.strip() else '<|en' if '<|en' in her_res.strip() else '<|e' if '<|e' in her_res.strip() else '<|' if '<|' in her_res.strip() else '<|endoftext|>'
            msglist = her_res.strip().split(x) 
            output+=(f"\n  {msglist[0]}")
    print("Her response:",output)
    return output
    
app = Flask(__name__)

@app.route("/bot", methods=["POST"])
def bot():
    msg = request.form.get('Body')
    sender = request.form.get('From')

    print(f"[{sender}]: {msg}")

    response_text = get_bot_reply(msg)

    reply = MessagingResponse()
    reply.message(response_text)

    return str(reply)

if __name__ == "__main__":
    app.run(port=5000)
