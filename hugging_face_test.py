import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

# generator = pipeline('text-generation', model='./DialoGPT-medium')
# print(generator('Tomorrow we will'))

tokenizer = AutoTokenizer.from_pretrained('./DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('./DialoGPT-medium')
sequence = 'What is a token?'
tokens = tokenizer.tokenize(sequence, return_tensors='pt')
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
output_ids = model.generate(torch.tensor([ids]), max_new_tokens=10, return_dict_in_generate=True)
print(output_ids)
print(tokenizer.decode(output_ids.sequences[0]))
