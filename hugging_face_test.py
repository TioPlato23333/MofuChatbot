from transformers import AutoTokenizer
from transformers import pipeline

# generator = pipeline('text-generation', model='./DialoGPT-medium')
# print(generator('Tomorrow we will'))

checkpoint = './DialoGPT-medium'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
raw_inputs = [
    'This is input 1',
    'Hello, I\'m the second input'
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
print(inputs)
