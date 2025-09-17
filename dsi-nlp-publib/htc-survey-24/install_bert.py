import os
from transformers import BertTokenizer, BertModel

model_name = "bert-base-uncased"
save_dir = "./bert-base-uncased-local"
os.makedirs(save_dir, exist_ok=True)

# Download and save tokenizer + model
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

model = BertModel.from_pretrained(model_name)
model.save_pretrained(save_dir)