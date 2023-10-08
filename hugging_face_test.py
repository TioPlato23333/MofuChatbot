from transformers import AutoTokenizer
from transformers import pipeline

# generator = pipeline('text-generation', model='./DialoGPT-medium')
# print(generator('Tomorrow we will'))

tokenizer = AutoTokenizer.from_pretrained('./DialoGPT-medium')
sequence = 'What is a token?'
tokens = tokenizer.tokenize(sequence)
print(tokens)