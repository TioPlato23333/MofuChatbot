from transformers import pipeline

generator = pipeline('text-generation', model='./DialoGPT-medium')
print(generator('Tomorrow we will'))
