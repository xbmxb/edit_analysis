import os, json, jsonlines
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import random
import argparse
import torch
import torch
import torch.nn.functional as F
from tqdm import tqdm
device = 'cuda:7'
tokenizer = LlamaTokenizer.from_pretrained("/data/maxb/tag/model_cache/llama-ckpt/llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("/data/maxb/tag/model_cache/llama-ckpt/llama-2-7b-chat-hf").to(device)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

def perplexity(model,tok,prompt: str,compl: str):
    inputs = tok(
        [prompt + compl], return_tensors="pt").to(device)
    input_token_len = tok(prompt, return_tensors="pt")["input_ids"].size(1)
    # print(prompt, compl, input_token_len, inputs["input_ids"].size(1))
    with torch.no_grad():
        logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]
    # log_probs = log_probs[input_token_len-1:]
    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / (inputs["input_ids"].size(1)-input_token_len) * log_probs.sum()).item()

with open('/data/maxb/mememe/EasyEdit/cf/counterfact.json', 'r') as f:
    data = json.load(f)
# data = data[:1000]
output = []
for i in tqdm(range(len(data))):
    ret_dic = {}
    request = data[i]['requested_rewrite']
    prompt = request['prompt'].format(request['subject'])
    target_true = ' ' + request['target_true']["str"]
    target_new = ' ' + request['target_new']["str"]
    target_p_true = perplexity(model, tokenizer, prompt, target_true)
    target_p_new = perplexity(model, tokenizer, prompt, target_new)
    ret_dic['case_id'] = data[i]['case_id']
    ret_dic['prompt'] = prompt
    ret_dic['true'] = target_true
    ret_dic['p_true'] = target_p_true
    ret_dic['new'] = target_new
    ret_dic['p_new'] = target_p_new
    # print(ret_dic)
    output.append(ret_dic)
with open('see_ppl_whole_setence.json', 'w') as f:
    json.dump(output, f, indent=2)
