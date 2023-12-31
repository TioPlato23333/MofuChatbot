from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

# generator = pipeline('text-generation', model='./DialoGPT-medium')
# print(generator('Tomorrow we will'))

'''
tokenizer = AutoTokenizer.from_pretrained('./DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('./DialoGPT-medium')
sequence = 'What is a token?'
tokens = tokenizer.tokenize(sequence + tokenizer.eos_token, return_tensors='pt')
print(tokens)
ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
print(ids)
output_ids = model.generate(ids, max_new_tokens=5, return_dict_in_generate=True, no_repeat_ngram_size=2)
print(output_ids)
print(ids.shape[-1])
print(tokenizer.decode(output_ids.sequences[0][ids.shape[-1]:]))
'''

if __name__ == '__main__':
    dataset = load_dataset('./lhoestq/demo1', split='train')
    print(dataset)
    for line in dataset:
        print(line['id'])
