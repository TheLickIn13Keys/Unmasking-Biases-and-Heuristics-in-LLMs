import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch

print(os.getcwd())
dataset = []
file_path = './content/LLMHeuristicReHEAT/test_cases_0320.jsonl'
with open(file_path, 'r') as file:
    for line in file:
        dataset.append(json.loads(line))

print(dataset[0])  # Check the first entry

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", output_attentions=True)

#model_name = "meta-llama/Meta-Llama-3-8B"
#tokenizer = LlamaTokenizer.from_pretrained(model_name)
#model = LlamaForCausalLM.from_pretrained(model_name, output_attentions=True)

# Set the model to evaluation mode
model.eval()

text_input = dataset[0]['test_case']  # Adjust this based on the structure of your dataset
inputs = tokenizer(text_input, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

if outputs.attentions is not None:
    attentions = outputs.attentions
    
    layer = 0  # First layer (0-indexed)
    head = 0   # First head (0-indexed)

    attention_matrix = attentions[layer][0, head].detach().numpy()  # (seq_len, seq_len)

    plt.imshow(attention_matrix, cmap='viridis')
    plt.title(f"Layer {layer + 1}, Head {head + 1}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.colorbar(label="Attention Weight")
    plt.show()
else:
    print("Attention weights were not returned by the model.")
