from transformers import AutoTokenizer, AutoModel
import torch


model_name = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Tokenizer aur Model load karo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "RAG is very useful for AI applications"

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)


embeddings = outputs.last_hidden_state.mean(dim=1)

print(embeddings.shape) 