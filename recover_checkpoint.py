from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the checkpoint
checkpoint_path = r"C:\MyEverything\PythonProjects\Recent_projects\cnn_analysis\YouAreMe\fine_tuned_ShirayukiV2_OPT\checkpoint-10040"

# Load the model and tokenizer from the checkpoint
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Save the recovered model to a new folder
model.save_pretrained("fine_tuned_ShirayukiV3_OPT")
tokenizer.save_pretrained("fine_tuned_ShirayukiV3_OPT")

print("Model successfully recovered and saved!")
