import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('DialoGPT-medium')
    model = AutoModelForCausalLM.from_pretrained('DialoGPT-medium')
    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(input('>> User:') + tokenizer.eos_token, \
            return_tensors='pt')
        # append the new user input tokens to the chat history
        if step == 0:
            bot_input_ids = new_user_input_ids
        else:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, \
            pad_token_id=tokenizer.eos_token_id)
        print(bot_input_ids)
        print(chat_history_ids)
        # pretty print last ouput tokens from bot
        print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], \
            skip_special_tokens=True)))
    sys.exit(0)
